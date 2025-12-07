# Splatt3r-SFM

extending splatt3r to n views using mast3r-sfm framework

### Setup:

setup the environment

```
conda env create -f environment.yml
pip install git+https://github.com/dcharatan/diff-gaussian-rasterization-modified
```

Complle CUDA Kernels for RoPE
```
cd src/mast3r_src/dust3r/croco/models/curope/
python setup.py build_ext --inplace
cd ../../../../../../
```

### instructions:

`python model_reeplacement_test.py`

to give paths to images, output dirs, etc, run `model_replacement_test.py` with the `--help` flag 

### Example:

```
python model_replacement_test.py \
  --images_path labtest_large \
  --output_dir /scratch/final_splats/ \
  --threshold_mode triangle \
  --output_filename rosin_lab.ply \
  --stride 1
```
### Flags:

```
  -h, --help            show this help message and exit
  --images_path IMAGES_PATH
                        Path to directory containing input images
  --output_dir OUTPUT_DIR
                        Output directory for results
  --threshold_mode {percentile,threshold,triangle}
                        Confidence threshold mode: "percentile", "threshold",
                        or "triangle" (Rosin method)
  --percentile_value PERCENTILE_VALUE
                        Percentile value (0-100) for confidence filtering when
                        using percentile mode
  --min_conf_threshold MIN_CONF_THRESHOLD
                        Fixed confidence threshold when using threshold mode
  --top_volume_percentile TOP_VOLUME_PERCENTILE
                        Remove top volume percentile to filter out large
                        floaters (0.0-1.0)
  --stride STRIDE       Stride for frame sampling. If not specified, all images
                        will be used
  --output_filename OUTPUT_FILENAME
                        Name of the output PLY file (default: final_splat.ply)
  --enable_logging      Enable logging to log.txt file
```

### Confidence Thresholding Types: 

1. Percentile - What percentile (by confidence) to keep. Use the `--percentile-value xx` flag along with `--threshold_mode percentile`
2. Threshold - Hard Confidence value to Threshold by. Use the `--min_conf_threshold xx` flag along with `--threshold_mode threshold`
3. Triangle(Rosin's Method) - This dynamically determines the best confidence threshold needed for your scene.

**Reference for Rosin's Method**: [Triangle Method](https://forum.image.sc/t/understanding-imagej-implementation-of-the-triangle-algorithm-for-threshold/752)


### Citations

```
@article{smart2024splatt3r,
      title={Splatt3R: Zero-shot Gaussian Splatting from Uncalibrated Image Pairs}, 
      author={Brandon Smart and Chuanxia Zheng and Iro Laina and Victor Adrian Prisacariu},
      year={2024},
      eprint={2408.13912},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.13912}, 
}

@misc{mast3r_eccv24,
      title={Grounding Image Matching in 3D with MASt3R}, 
      author={Vincent Leroy and Yohann Cabon and Jerome Revaud},
      booktitle = {ECCV},
      year = {2024}
}

@misc{mast3r_arxiv24,
      title={Grounding Image Matching in 3D with MASt3R}, 
      author={Vincent Leroy and Yohann Cabon and Jerome Revaud},
      year={2024},
      eprint={2406.09756},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@inproceedings{dust3r_cvpr24,
      title={DUSt3R: Geometric 3D Vision Made Easy}, 
      author={Shuzhe Wang and Vincent Leroy and Yohann Cabon and Boris Chidlovskii and Jerome Revaud},
      booktitle = {CVPR},
      year = {2024}
}

@inproceedings{
    duisterhof2025mastrsfm,
    title={{MAS}t3R-SfM: a Fully-Integrated Solution for Unconstrained Structure-from-Motion},
    author={Bardienus Pieter Duisterhof and Lojze Zust and Philippe Weinzaepfel and Vincent Leroy and Yohann Cabon and Jerome Revaud},
    booktitle={International Conference on 3D Vision 2025},
    year={2025},
    url={https://openreview.net/forum?id=5uw1GRBFoT}
} 
```
