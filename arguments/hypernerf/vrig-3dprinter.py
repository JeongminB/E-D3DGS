_base_ = './default.py'
ModelHiddenParams = dict(
    min_embeddings = 20,
    max_embeddings = 100,
    c2f_temporal_iter = 20000,
    total_num_frames = 207,
)

OptimizationParams = dict(
    maxtime = 207,
    iterations = 60_000,
    densify_until_iter = 60_000,
    position_lr_max_steps = 60_000,
    deformation_lr_max_steps = 60_000,
)