GPU=0
PORT_BASE=6000
GT_PATH=path

DATASET=hypernerf
SAVE_PATH=output

SCENE_LIST=(
    # vrig-3dprinter
    # vrig-broom
    vrig-chicken
    # vrig-peel-banana
)
for SCENE in "${SCENE_LIST[@]}"; do
    echo "scene: $SCENE"
    CONFIG=$SCENE
    CUDA_VISIBLE_DEVICES=$GPU python train.py -s $GT_PATH/$SCENE --port $(expr $PORT_BASE + $GPU) --model_path $SAVE_PATH/$DATASET/$CONFIG --expname $DATASET/$SCENE --configs arguments/$DATASET/$CONFIG.py -r 1
    CUDA_VISIBLE_DEVICES=$GPU python render.py --model_path $SAVE_PATH/$DATASET/$CONFIG  --skip_train --configs arguments/$DATASET/$CONFIG.py
    CUDA_VISIBLE_DEVICES=$GPU python metrics.py --model_path $SAVE_PATH/$DATASET/$CONFIG
done