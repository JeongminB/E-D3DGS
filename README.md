#  E-D3DGS : Embedding-Based Deformable 3D Gaussian Splatting (ECCV 2024)

[![arXiv](https://img.shields.io/badge/arXiv-2404.03613-006600)](https://arxiv.org/abs/2404.03613) 
[![project_page](https://img.shields.io/badge/project_page-68BC71)](https://jeongminb.github.io/e-d3dgs/)

[Jeongmin Bae](https://jeongminb.github.io/)<sup>1*</sup>, [Seoha Kim](https://seoha-kim.github.io/)<sup>1*</sup>, [Youngsik Yun](https://bbangsik13.github.io/)<sup>1</sup>, </br>
Hahyun Lee<sup>2 </sup>, Gun Bang<sup>2</sup>, [Youngjung Uh](https://github.com/yj-uh)<sup>1†</sup>

<sup>1</sup>Yonsei University &emsp; <sup>2</sup>Electronics and Telecommunications Research Institute (ETRI)
<br><sup>\*</sup> Equal Contributions &emsp; <sup>†</sup> Corresponding Author

---

Official repository for <a href="https://arxiv.org/abs/2404.03613">"Per-Gaussian Embedding-Based Deformation for Deformable 3D Gaussian Splatting"</a><be>. <br>
Our approach employs per-Gaussian latent embeddings to predict deformation for each Gaussian and achieves a clearer representation of dynamic motion.

![Alt Text](https://github.com/JeongminB/E-D3DGS/blob/main/teaser.gif)

## Environmental Setup
Please follow the [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) to install the relative packages.
```bash
git clone https://github.com/JeongminB/E-D3DGS.git
cd E-D3DGS
git submodule update --init --recursive

conda create -n ed3dgs python=3.7 
conda activate ed3dgs

# If submodules fail to be downloaded, refer to the repository of 3DGS  
pip install -r requirements.txt
pip install -e submodules/diff-gaussian-rasterization/
pip install -e submodules/simple-knn/ 
```
We use `pytorch=1.13.1+cu116` in our environment.


## Data Preparation

**Downloading Datasets:**  
Please download datasets from their official websites : [HyperNerf](https://github.com/google/hypernerf/releases/tag/v0.1), [Neural 3D Video](https://github.com/facebookresearch/Neural_3D_Video) and [Technicolor](https://www.interdigital.com/data_sets/light-field-dataset) <br><br>
- Note that 'cam13.mp4' and corresponding pose in <i>coffee_martini</i> scene on Neural 3D Dataset dataset should be mannually removed. <br>
- We split the entire <i>flame_salmon_1_split</i> scene into four 300-frame scenes.

<br>

**Extracting point clouds from COLMAP:** 
```bash
# setup COLMAP 
bash script/colmap_setup.sh
conda activate colmapenv 

# automatically extract the frames and reorginize them
python script/pre_n3v.py --videopath <dataset>/<scene>
python script/pre_technicolor.py --videopath <dataset>/<scene>
python script/pre_hypernerf.py --videopath <dataset>/<scene>

# downsample dense point clouds
python script/downsample_point.py \
<location>/<scene>/colmap/dense/workspace/fused.ply <location>/<scene>/points3D_downsample.ply
```


After running COLMAP, Neural 3D Video and Technicolor datasets are orginized as follows:
```
├── data
│   | n3v
│     ├── cook_spinach
│       ├── colmap
│       ├── images
│           ├── cam01
│               ├── 0000.png
│               ├── 0001.png
│               ├── ...
│           ├── cam02
│               ├── 0000.png
│               ├── 0001.png
│               ├── ...
│     ├── cut_roasted_beef
|     ├── ...
```

## Training

If you want to train with 2x downsampled images, add `-r 2` to the command line.
``` bash
# Train
python train.py -s $GT_PATH/$SCENE --configs arguments/$DATASET/$CONFIG.py --model_path $OUTPUT_PATH --expname $DATASET/$SCENE
``` 

## Rendering
The current code does not yet support video rendering for visualization (e.g., spiral path rendering).

``` bash
# Render test view only
python render.py --model_path $OUTPUT_PATH --configs arguments/$DATASET/$CONFIG.py --skip_train --skip_video

# Render train view only
python render.py --model_path $OUTPUT_PATH --configs arguments/$DATASET/$CONFIG.py --skip_test --skip_video
```

## Evaluation
Note: In our paper, we calculate FPS by measuring rendering time only (except for save_image, etc.).
``` bash
# Evaluate
python metrics.py --model_path $SAVE_PATH/$DATASET/$CONFIG
```

## Note

* We provide scripts that collectively perform training, rendering, and evaluation. See the `train_<dataset_name>.sh`. 
* You will need to configure the dataset path according to your system.
* In the config file, make sure that the `total_num_frames` and `maxtime` are equal to the total number of training frames.

## Acknowledgements

This code is based on [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), [4DGaussians](https://github.com/hustvl/4DGaussians) and [STG](https://github.com/oppo-us-research/SpacetimeGaussians). In particular, we used [4DGaussians](https://github.com/hustvl/4DGaussians) as a starting point for our study. We would like to thank the authors of these papers for their hard work. 😊

## BibTex
```
@inproceedings{bae2024ed3dgs,
    title={Per-Gaussian Embedding-Based Deformation for Deformable 3D Gaussian Splatting}, 
    author={Bae, Jeongmin and Kim, Seoha and Yun, Youngsik and Lee, Hahyun and Bang, Gun and Uh, Youngjung}, 
    booktitle = {European Conference on Computer Vision (ECCV)},
    year={2024}
}
```
