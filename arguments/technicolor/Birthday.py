_base_ = './default.py'
OptimizationParams = dict(
    iterations = 100_000,
    densify_until_iter_coarse = 100_000,
    densify_until_iter = 100_000,
    position_lr_max_steps_coarse = 100_000,
    position_lr_max_steps = 100_000,
    deformation_lr_max_steps = 100_000,
    reg_coef=0.1
)