GPU=0
PORT_BASE=6000
GT_PATH=/home/jmin/nas_jmin/2024_E-D3DGS/e-d3dgs_colmap_test

DATASET=dynerf
SAVE_PATH=output

SCENE_LIST=(
    # coffee_martini_wo_cam13
    # cook_spinach
    # cut_roasted_beef
    # flame_salmon_frag1
    # flame_salmon_frag2
    # flame_salmon_frag3
    # flame_salmon_frag4
    # flame_steak
    sear_steak
)
for SCENE in "${SCENE_LIST[@]}"; do
    echo "scene: $SCENE"
    CONFIG=$SCENE
    CUDA_VISIBLE_DEVICES=$GPU python train.py -s $GT_PATH/$SCENE --port $(expr $PORT_BASE + $GPU) --model_path $SAVE_PATH/$DATASET/$CONFIG --expname $DATASET/$SCENE --configs arguments/$DATASET/$CONFIG.py -r 2
    CUDA_VISIBLE_DEVICES=$GPU python render.py --model_path $SAVE_PATH/$DATASET/$CONFIG  --skip_train --configs arguments/$DATASET/$CONFIG.py
    CUDA_VISIBLE_DEVICES=$GPU python metrics.py --model_path $SAVE_PATH/$DATASET/$CONFIG
done