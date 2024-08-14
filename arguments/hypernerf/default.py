ModelParams = dict(
    loader = "nerfies",
    shuffle = False
)

ModelHiddenParams = dict(
    defor_depth = 0,
    net_width = 128,
    no_ds = False,
    no_do = True,
    no_dc = True,
    
    temporal_embedding_dim = 256,
    gaussian_embedding_dim = 32,
    use_coarse_temporal_embedding = True,
    zero_temporal = True,
    use_anneal = False,
)

OptimizationParams = dict(
    dataloader = True,
    opacity_reset_interval = 6000000,
    densify_from_iter_coarse = 500,
    densify_from_iter = 500,    
    pruning_from_iter = 500,
    densification_interval = 100,

    densify_grad_threshold_fine_init = 0.0002,
    densify_grad_threshold_after = 0.0002,
    opacity_threshold_fine_init = 0.005,
    opacity_threshold_fine_after = 0.005,
    
    position_lr_init = 0.00016,
    position_lr_final = 0.0000016,
    position_lr_delay_mult = 0.01,

    deformation_lr_init = 0.00016,
    deformation_lr_final = 0.000016,
    deformation_lr_delay_mult = 0.01,
    deformation_lr_max_steps = 80_000,

    feature_lr = 0.0025,
    feature_lr_div_factor = 20.0,
    opacity_lr = 0.05,
    scaling_lr = 0.005,
    rotation_lr = 0.001,
    # pruning_interval = 2000
    
    scene_bbox_min = [-3.0, -1.8, -1.2],
    scene_bbox_max = [3.0, 1.8, 1.2],
    num_pts = 2000,
    threshold = 3,
    downsample = 1.0,

    coarse_stage_frame_num = 0,
    lambda_dssim = 0,
    num_multiview_ssim = 0,
    use_colmap = True,
    offsets_lr = 0,

    coef_tv_temporal_embedding = 0.0001,
    reg_coef = 1,
)