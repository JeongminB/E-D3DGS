ModelParams = dict(
    points_type = "downsample",
    loader= "technicolor"
)

ModelHiddenParams = dict(
    defor_depth = 0,
    net_width = 128,
    no_ds = False,
    no_do = False,
    no_dc = True,
    
    min_embeddings = 5,
    max_embeddings = 25,
    total_num_frames = 50,
    temporal_embedding_dim = 256,
    gaussian_embedding_dim = 32,
    use_coarse_temporal_embedding = True,
    no_c2f_temporal_embedding = True,
    
    c2f_temporal_iter = 10000,
    use_anneal = False,
    zero_temporal = True,
)

OptimizationParams = dict(
    dataloader=True,
    iterations = 80_000,
    maxtime = 50,

    densify_until_iter = 80_000,
    opacity_reset_interval = 6000000,

    densify_from_iter = 500,    
    pruning_from_iter = 500,
    densification_interval = 100,

    densify_grad_threshold_fine_init = 0.0003,
    densify_grad_threshold_after = 0.0003,
    
    opacity_threshold_fine_init = 0.005,
    opacity_threshold_fine_after = 0.005,
    
    position_lr_init = 0.00016,
    position_lr_final = 0.0000016,
    position_lr_delay_mult = 0.01,
    position_lr_max_steps = 80_000,

    deformation_lr_init = 0.0016,
    deformation_lr_final = 0.00016,
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

    lambda_dssim = 1,
    ssim_temperature = 1,
    num_multiview_ssim = 5,
    use_colmap=True,
    offsets_lr = 0,
    random_until = 60000,
)