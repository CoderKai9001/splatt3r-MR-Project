import torch
import os
import numpy as np
from tqdm import tqdm
import sys

sys.path.append('src/mast3r_src')
sys.path.append('src/mast3r_src/dust3r')
sys.path.append('src/pixelsplat_src')

from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
from mast3r.utils.misc import hash_md5
from dust3r.inference import inference
from PIL import Image
from typing import Tuple, Union, Optional
from pathlib import Path
import torchvision.transforms as tfm
from natsort import natsorted
from mast3r.image_pairs import make_pairs
# from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from mast3r.cloud_opt.new_sparse_ga import sparse_global_alignment
from dust3r.utils.image import load_images
from huggingface_hub import hf_hub_download
import utils.geometry as geometry
from main import MAST3RGaussians
from src.mast3r_src.demo import get_3D_model_from_scene
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation

class MASt3R:
    def __init__(self, imgdir, outdir):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.H=512
        self.W=512
        self.imgdir = imgdir
        self.outdir = outdir
        self.imgNames = natsorted(os.listdir(f'{imgdir}'))
        self.imgNames = [f'{imgdir}/{imgName}' for imgName in self.imgNames]
        self.npzFullPath = f"{outdir}/nodes_mast3r_points.npz"
        self.normalize = tfm.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # self.model_path = Path("checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth")
        # self.model = AsymmetricMASt3R.from_pretrained(self.model_path).to(self.device)
        # self.splatt3r_model = None
        try:
            model_name = "brandonsmart/splatt3r_v1.0"
            filename = "epoch=19-step=1200.ckpt"
            weights_path = hf_hub_download(repo_id=model_name, filename=filename)
            self.model = MAST3RGaussians.load_from_checkpoint(weights_path, device=self.device)
            self.model.to(self.device)  # Ensure model is on correct device
            print("Successfully loaded Splatt3r model")
        except Exception as e:
            print(f"Could not load Splatt3r model: {e}")
            print("Continuing without Splatt3r model...")
            self.model = None

    def preprocess_image(self, img):
        """
        Applies pre-processing transformations to the image. 
        """
        _, h, w = img.shape
        orig_shape = h, w

        # Normalize the image
        img = self.normalize(img).unsqueeze(0)

        return img, orig_shape

    def load_single_image(self, path: Union[str, Path], resize: Optional[Union[int, Tuple]] = None, rot_angle: float = 0) -> torch.Tensor:
        """
        Loads a single image and resizes it to the pairs = make_pairs(imgs, scene_graph=scene_graph, prefilter=None, symmetrize=True) Height and Width specified in the config.
        """
        if isinstance(resize, int):
            resize = (resize, resize)
        if isinstance(path, str):
            path = Path(path)
            img = Image.open(path).convert("RGB")
        else:
            img = path
        img = tfm.ToTensor()(img)
        if resize is not None:
            img = tfm.Resize(resize, antialias=True)(img)
        img = tfm.functional.rotate(img, rot_angle)
        return img
    
    def load_images(self, img0_path, img1_path):
        """
        Loads and calls pre-processing to get the images ready for mast3r inference
        """
        img0 = self.load_single_image(img0_path, (self.H, self.W))
        img1 = self.load_single_image(img1_path, (self.H, self.W))

        img0, img0_orig_shape = self.preprocess_image(img0)
        img1, img1_orig_shape = self.preprocess_image(img1)

        img_pair = [
            {"img": img0, "idx": 0, "instance": 0, "true_shape": np.int32([img0.shape[-2:]])},
            {"img": img1, "idx": 1, "instance": 1, "true_shape": np.int32([img1.shape[-2:]])},
        ]

        return img_pair

    def infer_mast3r(self, img0_path, img1_path):
        """
        Gets 3d pointcloud predictions for given 2 images.
        """
        images = self.load_images(img0_path, img1_path)
        output = inference([tuple(images)], self.model, self.device, batch_size=1, verbose=False)
        self.output = output
        return output

    def create_points_npz(self, save_points=True):
        """
        Given images in a dir, gets the 3d pointcloud predictions for each image and 
        Saves them(optional)
        """
        pc_dict = {}
        for i, _ in enumerate(tqdm(self.imgNames, desc="Getting Mast3r Points")):
            cur_img_path = self.imgNames[i]

            output = self.infer_mast3r(cur_img_path, cur_img_path)
            pred1 = output["pred1"] 
            pts3d = pred1['pts3d'][0]

            pc_dict[cur_img_path] = pts3d
        
        if save_points:
            np.savez_compressed(self.npzFullPath, **pc_dict)
            print(f"Saved {len(pc_dict)} point clouds to {self.npzFullPath}")
        
        return pc_dict

    def get_mast3r_matches(self, img0_path, img1_path):
        """
        Gets Mast3r matches for given images
        """
        images = self.load_images(img0_path, img1_path)
        output = inference([tuple(images)], self.model, self.device, batch_size=1, verbose=False)

        # at this stage, you have the raw dust3r predictions
        view1, pred1 = output['view1'], output['pred1']
        view2, pred2 = output['view2'], output['pred2']

        desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()

        # find 2D-2D matches between the two images
        matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8,
                                                    device=self.device, dist='dot', block_size=2**13)

        # ignore small border around the edge
        H0, W0 = view1['true_shape'][0]
        valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) & (
            matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3)

        H1, W1 = view2['true_shape'][0]
        valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) & (
            matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)

        valid_matches = valid_matches_im0 & valid_matches_im1
        matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]

        # (num_matches, 2)
        return matches_im0, matches_im1

    def reconstruct_scene(self,
                          outdir: str,
                          cache_dir: str = "cache_mast3r",
                          scene_graph: str = "complete",
                          optim_level: str = "refine+depth",
                          lr1: float = 0.07, niter1: int = 300,
                          lr2: float = 0.01, niter2: int = 300,
                          matching_conf_thr: float = 0.0,
                          shared_intrinsics: bool = False,
                          silent: bool = False,
                          **kw):
        """
        Run full Mast3R SfM (pairwise inference -> make_pairs -> sparse_global_alignment).
        Returns the scene object produced by sparse_global_alignment.
        """
        # filelist: ordered list of image paths
        filelist = self.imgNames

        # load resized images (used by make_pairs heuristics)
        imgs = load_images(filelist, size=self.H, verbose=not silent)

        print(f"----Using scene graph method {scene_graph}---")
        # build pairs according to scene_graph strategy
        pairs = make_pairs(imgs, scene_graph=scene_graph, prefilter=None, symmetrize=True)

        print(f"{len(pairs)} pairs constructed")

        os.makedirs(cache_dir, exist_ok=True)

        scene = sparse_global_alignment(filelist, pairs, cache_dir,
                                        self.model,
                                        lr1=lr1, niter1=niter1,
                                        lr2=lr2, niter2=niter2,
                                        device=self.device,
                                        opt_depth='depth' in optim_level,
                                        shared_intrinsics=shared_intrinsics,
                                        matching_conf_thr=matching_conf_thr,
                                        subsample=8,
                                        **kw)
        return scene
    
    def retrieve_gaussians(self, scene, cache_dir="mast3r_cache"):
        """
        Get Gaussian Attributes from cache dir
        """
        image_hashes = []
        for i, img_path in enumerate(scene.img_paths):
            img_hash = hash_md5(img_path)
            image_hashes.append((i, img_path, img_hash))
            print(f"Image {i}: {img_path} -> {img_hash}")

        gaussian_attributes = {}

        for img_idx, img_path, img_hash in image_hashes:
            # gaussians_path = f"{cache_dir}/gaussian_attributes/{img_hash}.pth"
            gaussians_path = f"{cache_dir}/canon_gaussians/{img_hash}_subsample=8.pth"

            try:
                # Load the Gaussian attributes for this image
                gaussians = torch.load(gaussians_path)
                # sh, scales, rotations, opacities, means = gaussians
                
                gaussian_attributes[img_idx] = {
                    'image_path': img_path,
                    'hash': img_hash,
                    'sh': gaussians['sh'],
                    'scales': gaussians['scales'], 
                    'rotations': gaussians['rotations'],
                    'opacities': gaussians['opacities'],
                    'means': gaussians['means'],
                    'hash': img_hash
                }
                
                print(f"Loaded Gaussians for image {img_idx}: {gaussians['sh'].shape}")
                
            except FileNotFoundError:
                print(f"Gaussian attributes not found for image {img_idx} (hash: {img_hash})")

        return gaussian_attributes

    def get_pts_to_gaussian_map(self, scene, gaussian_attributes):
        """
        Memory-safe build of Gaussian attributes for all dense pts3d in scene.
        Returns a dict of numpy arrays:
        {
            'pts3d': (N,3),
            'pixels': (N,2),   # x,y
            'image_index': (N,),
            'color': (N,3),
            'sh': (N, K_sh),
            'scales': (N,3),
            'rotations': (N,4) or (N,3) depending on source,
            'opacities': (N,1),
            'means': (N,3),
            'covariances': (N,3,3)  # or (N,9) flattened
        }
        """
        pts3d_dense, _, _ = scene.get_dense_pts3d(clean_depth=True)

        # Lists to collect final attributes
        pts3d_list = []
        pixels_list = []
        imgidx_list = []
        color_list = []
        sh_list = []
        scales_list = []
        rot_list = []
        opac_list = []
        means_list = []
        cov_list = []

        # Iterate per image and process entire attribute tensors once
        for img_idx, pts3d_img in enumerate(tqdm(pts3d_dense, desc="Processing images", unit="img")):
            # Convert gaussian attributes for this image to CPU numpy ONCE
            if img_idx not in gaussian_attributes:
                # skip if attributes missing
                continue
            attr = gaussian_attributes[img_idx]

            # Convert full tensors to numpy on CPU once
            # Handle both torch.Tensor and numpy already
            def to_np(x):
                if isinstance(x, torch.Tensor):
                    return x.detach().cpu().numpy()
                return np.asarray(x)

            sh_np = to_np(attr['sh'])          # expected shape (1, H, W, K_sh) 
            scales_np = to_np(attr['scales'])  # (1, H, W, 3)
            rot_np = to_np(attr['rotations'])  # (1, H, W, 4) or (1, H, W, 3)
            op_np = to_np(attr['opacities'])   # (1, H, W, 1) or (1, H, W)
            means_np = to_np(attr['means'])    # (1, H, W, 3)

            # Remove the batch dimension (squeeze first axis)
            sh_np = sh_np.squeeze()          # (H, W, K_sh)
            scales_np = scales_np.squeeze()  # (H, W, 3)
            rot_np = rot_np.squeeze()        # (H, W, 4) or (H, W, 3)
            op_np = op_np.squeeze()          # (H, W, 1) or (H, W)
            means_np = means_np.squeeze()    # (H, W, 3)

            # Build covariance map ONCE per image:
            # If build_covariance returns per-pixel HxWx3x3, convert to numpy.
            cov_map = geometry.build_covariance(attr['scales'], attr['rotations'])
            if isinstance(cov_map, torch.Tensor):
                cov_map = cov_map.detach().cpu().numpy()  # (1, H, W, 3, 3) ideally
                cov_map = cov_map.squeeze()  # Remove batch dimension -> (H, W, 3, 3)

            # Access the image pixels (scene.imgs likely already numpy or torch)
            img = scene.imgs[img_idx]
            if isinstance(img, torch.Tensor):
                img_np = img.detach().cpu().numpy()
            else:
                img_np = np.asarray(img)

            H = self.H
            W = self.W
            # safe-check shapes: if sh/scales dims have extra leading dims, squeeze appropriately
            # We'll index as [y, x, ...] so ensure shapes are (H, W, ...)
            # Now iterate pixels (only where there is a dense point)
            for y in range(H):
                for x in range(W):
                    linear_idx = y * W + x
                    if linear_idx >= len(pts3d_img):
                        continue
                    pt3d = pts3d_img[linear_idx]
                    # Convert pt3d to numpy once per point
                    if isinstance(pt3d, torch.Tensor):
                        pt3d_np = pt3d.detach().cpu().numpy()
                    else:
                        pt3d_np = np.asarray(pt3d)

                    # color from image (img_np expected shape [C,H,W] or [H,W,C])
                    if img_np.ndim == 3 and img_np.shape[0] == 3:
                        # CHW -> HWC
                        color_px = img_np[:, y, x].astype(np.float32)
                        color_px = color_px[::-1] if False else color_px  # optional
                    else:
                        # already HWC
                        color_px = img_np[y, x].astype(np.float32)

                    pts3d_list.append(pt3d_np.astype(np.float32))
                    pixels_list.append((int(x), int(y)))
                    imgidx_list.append(int(img_idx))
                    color_list.append(color_px)

                    # index numpy attribute arrays; use safe indexing even if shapes differ
                    sh_list.append(np.asarray(sh_np[y, x]).astype(np.float32))
                    scales_list.append(np.asarray(scales_np[y, x]).astype(np.float32))
                    rot_list.append(np.asarray(rot_np[y, x]).astype(np.float32))
                    op_val = np.asarray(op_np[y, x])
                    op_list_val = op_val.item() if np.isscalar(op_val) else op_val.astype(np.float32)
                    opac_list.append(np.asarray(op_list_val))
                    means_list.append(np.asarray(means_np[y, x]).astype(np.float32))

                    # cov_map[y,x] expected (3,3); if shape mismatch, try flatten or compute fallback
                    cov_px = cov_map[y, x] if cov_map is not None else np.eye(3, dtype=np.float32)
                    cov_list.append(np.asarray(cov_px).astype(np.float32))

        # Stack lists into arrays
        pts3d_arr = np.stack(pts3d_list, axis=0) if pts3d_list else np.zeros((0,3), dtype=np.float32)
        pixels_arr = np.array(pixels_list, dtype=np.int32) if pixels_list else np.zeros((0,2), dtype=np.int32)
        imgidx_arr = np.array(imgidx_list, dtype=np.int32) if imgidx_list else np.zeros((0,), dtype=np.int32)
        color_arr = np.stack(color_list, axis=0) if color_list else np.zeros((0,3), dtype=np.float32)
        sh_arr = np.stack(sh_list, axis=0) if sh_list else np.zeros((0,3), dtype=np.float32)
        scales_arr = np.stack(scales_list, axis=0) if scales_list else np.zeros((0,3), dtype=np.float32)
        rot_arr = np.stack(rot_list, axis=0) if rot_list else np.zeros((0,4), dtype=np.float32)
        op_arr = np.stack(opac_list, axis=0).reshape(-1, 1) if opac_list else np.zeros((0,1), dtype=np.float32)
        means_arr = np.stack(means_list, axis=0) if means_list else np.zeros((0,3), dtype=np.float32)
        cov_arr = np.stack(cov_list, axis=0) if cov_list else np.zeros((0,3,3), dtype=np.float32)

        out = {
            'pts3d': pts3d_arr,
            'pixels': pixels_arr,
            'image_index': imgidx_arr,
            'color': color_arr,
            'sh': sh_arr,
            'scales': scales_arr,
            'rotations': rot_arr,
            'opacities': op_arr,
            'means': means_arr,
            'covariances': cov_arr
        }
        print(f"Created mapping for {pts3d_arr.shape[0]} Gaussian points")
        return out


    def save_gaussians_as_ply(self, coords_to_gaussians_map, save_path):
        """
        Fully safe version – no .detach() on NumPy, no GPU tensors left,
        converts everything to NumPy before building the PLY array.
        """
        import numpy as np
        from plyfile import PlyData, PlyElement
        from scipy.spatial.transform import Rotation

        def construct_list_of_attributes():
            return [
                "x", "y", "z",
                "nx", "ny", "nz",
                "f_dc_0", "f_dc_1", "f_dc_2",
                "opacity",
                "scale_0", "scale_1", "scale_2",
                "rot_0", "rot_1", "rot_2", "rot_3",
            ]

        def rgb_to_sh0(rgb):
            C0 = 0.28209479177387814
            return (rgb - 0.5) / C0

        def covariance_to_quat_and_scale(covs_np):
            """
            covs_np: (N, 3, 3) NumPy array
            Returns quaternions + scales as NumPy arrays.
            """
            # Convert back to torch for SVD (more stable than eigendecomposition)
            covs_torch = torch.from_numpy(covs_np)
            
            # Use SVD like the original working version
            U, S, V = torch.linalg.svd(covs_torch)
            scale = torch.sqrt(S).detach().cpu().numpy()
            rotation_matrix = torch.bmm(U, V.transpose(-2, -1))
            rotation_matrix_np = rotation_matrix.detach().cpu().numpy()
            
            # Convert to quaternions
            rotation = Rotation.from_matrix(rotation_matrix_np)
            quaternion = rotation.as_quat()
            
            return quaternion, scale

        # --------------------------------------------
        # 1. Aggregate all Gaussian data
        # --------------------------------------------
        means_list = []
        colors_list = []
        opacities_list = []
        covs_list = []

        print("Collecting Gaussian attributes...")
        for idx, g in tqdm(coords_to_gaussians_map.items(), desc="Collecting", unit="gaussian"):

            # ensure pt_3d is numpy
            pt = g['pt_3d']
            if isinstance(pt, torch.Tensor):
                pt = pt.detach().cpu().numpy()
            means_list.append(pt)

            # color
            col = g['color']
            if isinstance(col, torch.Tensor):
                col = col.detach().cpu().numpy()
            colors_list.append(col)

            # opacity
            op = g['opacities']
            if isinstance(op, torch.Tensor):
                op = op.detach().cpu().numpy()
            opacities_list.append(op)

            # covariance (ALREADY numpy → no detach/cpu!!!)
            cov = g['covariances']
            covs_list.append(cov)

        means = np.array(means_list).reshape(-1, 3)
        colors = np.array(colors_list).reshape(-1, 3)
        opacities = np.array(opacities_list).reshape(-1, 1)
        covs = np.array(covs_list).reshape(-1, 3, 3)

        # --------------------------------------------
        # 2. Convert colors → SH0
        # --------------------------------------------
        sh_dc = rgb_to_sh0(colors)

        # --------------------------------------------
        # 3. Covariance → Quaternion + Scale
        # --------------------------------------------
        print("Converting covariance → quaternion + scale...")
        quats, scales = covariance_to_quat_and_scale(covs)
        scales = np.log(scales)
        # --------------------------------------------
        # 4. Build the structured array
        # --------------------------------------------
        attrs = construct_list_of_attributes()
        dtype = [(name, "f4") for name in attrs]

        N = means.shape[0]
        elements = np.empty(N, dtype=dtype)

        print("Building final PLY structured array...")
        for i in tqdm(range(N), desc="Writing array"):
            elements[i] = (
                means[i, 0], means[i, 1], means[i, 2],       # xyz
                0.0, 0.0, 0.0,                                # normals placeholder
                sh_dc[i, 0], sh_dc[i, 1], sh_dc[i, 2],        # SH DC (RGB)
                float(opacities[i]),                          # opacity
                scales[i, 0], scales[i, 1], scales[i, 2],     # scales
                quats[i, 0], quats[i, 1], quats[i, 2], quats[i, 3],  # rotation quaternion
            )

        # --------------------------------------------
        # 5. Write PLY
        # --------------------------------------------
        print(f"Saving PLY to: {save_path}")
        vertex_el = PlyElement.describe(elements, "vertex")
        PlyData([vertex_el]).write(save_path)
        print(f"✓ Saved {N} Gaussians to {save_path}")


    def save_pointcloud_with_color(self, scene, out_ply: Union[str, Path]):
        """
        Save a colored pointcloud (PLY ascii) from a reconstructed scene.
        Expects scene to expose scene.pts3d and optionally scene.pts3d_colors.
        Handles list/tuple/dict/numpy-array/torch.Tensor types for pts and colors.
        """
        out_ply = str(out_ply)

        def _to_numpy(x):
            # convert torch tensor or numpy-like to numpy on CPU
            if x is None:
                return None
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            return np.asarray(x)

        def _concat_items(x):
            if x is None:
                return None
            arrs = []
            if isinstance(x, dict):
                items = x.values()
            elif isinstance(x, (list, tuple)):
                items = x
            else:
                arr = _to_numpy(x)
                if arr.size == 0:
                    return np.zeros((0, 3))
                return arr.reshape(-1, 3)
            for v in items:
                arr = _to_numpy(v)
                if arr is None or arr.size == 0:
                    continue
                arrs.append(arr.reshape(-1, 3))
            return np.concatenate(arrs, axis=0) if arrs else np.zeros((0, 3))

        pts = None
        if hasattr(scene, "pts3d"):
            pts = scene.pts3d
        elif hasattr(scene, "points"):
            pts = scene.points
        pts_all = _concat_items(pts)

        if pts_all is None or pts_all.size == 0:
            raise ValueError("No points found in scene (scene.pts3d / scene.points)")

        cols_all = None
        if hasattr(scene, "pts3d_colors"):
            cols_all = _concat_items(scene.pts3d_colors)
        elif hasattr(scene, "pts3d_color"):
            cols_all = _concat_items(scene.pts3d_color)

        # fallback to white if no colors available
        if cols_all is None or cols_all.size == 0:
            cols_all = np.tile(np.array([255, 255, 255], dtype=np.uint8), (pts_all.shape[0], 1))
        else:
            # if color count doesn't match point count, try to broadcast or trim
            if cols_all.shape[0] != pts_all.shape[0]:
                if cols_all.shape[0] < pts_all.shape[0]:
                    # repeat last color
                    pad = np.tile(cols_all[-1:], (pts_all.shape[0] - cols_all.shape[0], 1))
                    cols_all = np.concatenate([cols_all, pad], axis=0)
                else:
                    cols_all = cols_all[:pts_all.shape[0]]
            # normalize float colors to 0-255
            if cols_all.dtype.kind == "f":
                cols_all = (np.clip(cols_all, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                cols_all = cols_all.astype(np.uint8)

        # write PLY ASCII
        os.makedirs(os.path.dirname(out_ply) or ".", exist_ok=True)
        n = int(pts_all.shape[0])
        with open(out_ply, "w") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {n}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            for i in range(n):
                x, y, z = pts_all[i]
                r, g, b = cols_all[i]
                f.write(f"{float(x)} {float(y)} {float(z)} {int(r)} {int(g)} {int(b)}\n")
        return out_ply

if __name__ == "__main__":
    gen3d = MASt3R(imgdir=Path("./images"), outdir=Path("./pointclouds"))

    # run multi-view Mast3R SfM
    print("running Mast3r-sfm")
    scene = gen3d.reconstruct_scene(outdir=str(gen3d.outdir),
                                    cache_dir="mast3r_cache",
                                    scene_graph="swin-5-noncyclic",
                                    optim_level="refine+depth",
                                    lr1=0.07, niter1=300,
                                    lr2=0.01, niter2=300)
    
    print("SFM Done!")

    gaussian_attributes = gen3d.retrieve_gaussians(scene, cache_dir="mast3r_cache")
    print("Retrieved Gaussian attrs from cache\nGetting 3dpts to gaussians map")
    coords_to_gaussians_map = gen3d.get_pts_to_gaussian_map(scene, gaussian_attributes)
    print("map obtained!\nSaving final splat")
    
    # Convert the data format to what save_gaussians_as_ply expects
    gaussian_dict = {}
    for i in range(len(coords_to_gaussians_map['pts3d'])):
        gaussian_dict[i] = {
            'pt_3d': coords_to_gaussians_map['pts3d'][i],
            'color': coords_to_gaussians_map['color'][i], 
            'opacities': coords_to_gaussians_map['opacities'][i],
            'covariances': coords_to_gaussians_map['covariances'][i]
        }
    
    gen3d.save_gaussians_as_ply(gaussian_dict, "pointclouds/canon_averaged.ply")
    # gen3d.save_pointcloud_with_color(scene, out_ply="pointclouds/pcd_ver.ply")
    print("Saved Splat!\n All Done!!")
