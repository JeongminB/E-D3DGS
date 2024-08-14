_base_ = './default.py'
ModelHiddenParams = dict(
    min_embeddings = 16,
    max_embeddings = 80,
    c2f_temporal_iter = 10000,
    total_num_frames = 164,
)

OptimizationParams = dict(
    maxtime = 164,
    iterations = 60_000,
    densify_until_iter = 60_000,
    position_lr_max_steps = 60_000,
    deformation_lr_max_steps = 60_000,
)