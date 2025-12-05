import torch
import os
import numpy as np
from tqdm import tqdm
import sys

sys.path.append('src/mast3r_src')
sys.path.append('src/mast3r_src/dust3r')

from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
from dust3r.inference import inference
from PIL import Image
from typing import Tuple, Union, Optional
from pathlib import Path
import torchvision.transforms as tfm
from natsort import natsorted
from mast3r.image_pairs import make_pairs
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from dust3r.utils.image import load_images

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
        self.model_path = Path("checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth")
        self.model = AsymmetricMASt3R.from_pretrained(self.model_path).to(self.device)

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
                                        **kw)
        return scene
    
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
    gen3d = MASt3R(imgdir=Path("./airport"), outdir=Path("./pointclouds"))

    # run multi-view Mast3R SfM
    scene = gen3d.reconstruct_scene(outdir=str(gen3d.outdir),
                                    cache_dir="mast3r_cache",
                                    scene_graph="swin-4-noncyclic",
                                    optim_level="refine+depth",
                                    lr1=0.07, niter1=200,
                                    lr2=0.01, niter2=200)
    print("Reconstruction finished. Scene object:", type(scene))
    # access scene fields (example)
    try:
        # print("num images:", len(scene.imgs))
        # print(f"depthmaps: {scene.depthmaps}")
        # print(f"pts3d: {type(scene.pts3d)}")
        # print(f"{scene.pts3d.shape}")
        # print(f"pts3d_colors: {scene.pts3d_colors}")
        out_ply = Path(gen3d.outdir) / "airport_pcd.ply"
        saved = gen3d.save_pointcloud_with_color(scene, out_ply)
        print("Saved colored pointcloud to:", saved)
    except Exception as e:
        print(f"error: {e}")