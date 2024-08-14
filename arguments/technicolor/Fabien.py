_base_ = './default.py'
OptimizationParams = dict(
    iterations = 120_000,
    densify_until_iter_coarse = 120_000,
    densify_until_iter = 120_000,
    position_lr_max_steps_coarse = 120_000,
    position_lr_max_steps = 120_000,
    deformation_lr_max_steps = 120_000,
    reg_coef=0.1
)