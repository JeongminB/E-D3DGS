_base_ = './default.py'
OptimizationParams = dict(
    deformation_lr_init = 0.005,
    deformation_lr_final = 0.00005,
    deformation_lr_delay_mult = 0.01,
    
    iterations = 120_000,
    densify_until_iter_coarse = 120_000,
    densify_until_iter = 120_000,
    position_lr_max_steps_coarse = 120_000,
    position_lr_max_steps = 120_000,
    deformation_lr_max_steps = 120_000,
    reg_coef=0.1
)