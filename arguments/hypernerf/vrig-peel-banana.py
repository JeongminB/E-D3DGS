_base_ = './default.py'
ModelHiddenParams = dict(
    min_embeddings = 50,
    max_embeddings = 250,
    c2f_temporal_iter = 40000,
    total_num_frames = 513,
)

OptimizationParams = dict(
    maxtime = 513,
    iterations = 80_000,
    densify_until_iter = 80_000,
    position_lr_max_steps = 80_000,
    deformation_lr_max_steps = 80_000,
)