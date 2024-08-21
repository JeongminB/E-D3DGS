#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import math
import numpy as np
import random
import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss, lpips_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from utils.timer import Timer
from utils.extra_utils import o3d_knn, weighted_l2_loss_v2, image_sampler, calculate_distances, sample_camera

# import lpips
from utils.scene_utils import render_training_image
from time import time
to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)


def scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations, 
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, tb_writer, train_iter,timer, start_time):
    first_iter = 0

    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0

    final_iter = train_iter
    
    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter += 1

    train_cams = scene.getTrainCameras()
    test_cams = scene.getTestCameras()
    video_cams = None

    num_traincams = 1
    if dataset.loader != 'nerfies': # for multi-view setting
        num_traincams = int(len(train_cams) / scene.maxtime)
    
        camera_centers = []
        for i in range(num_traincams):
            camera_centers.append(train_cams[i*scene.maxtime].camera_center.cpu().numpy())
        camera_centers = np.array(camera_centers)
        cam_dists = calculate_distances(camera_centers)
        sorted_dists = np.unique(cam_dists)
        min_dist = sorted_dists[int(sorted_dists.shape[0] * 0.5)]

        last_camera_index = 0
    
    cam_no_list = list(set(c.cam_no for c in train_cams))
    print("train cameras:", cam_no_list)
    if dataset.loader in ['nerfies']:  # single-view
        loss_list = np.zeros([num_traincams, scene.maxtime]) + 100  # pick frames that have not yet been sampled
    else:  # n3v, technicolor, etc.
        loss_list = np.zeros([max(cam_no_list) + 1, scene.maxtime])
        for c in cam_no_list:
            loss_list[c] = 100

    ssim_cnt = 0
    sampled_frame_no = None
    prev_num_pts = 0

    # We sort training images to sample image of the desired camera number and frame.
    if dataset.loader not in ['nerfies']:
        train_cams = sorted(train_cams, key=lambda x: (x.cam_no, x.frame_no))

    viewpoint_stack = train_cams
    method = None

    start_time = time()
    for iteration in range(first_iter, final_iter+1):             
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # opt.batch_size = 2
        ### Instead of the complex process below, simply training on random frames will also work well. If you follow this, comment out the `train_cams` sorting process above.
        if dataset.loader == 'nerfies':
            frame_set = np.random.choice(range(math.ceil(len(viewpoint_stack) / 2)), size=max(opt.batch_size // 2, 1))
            viewpoint_cams = [viewpoint_stack[(f*2) % scene.maxtime] for f in frame_set] + \
                             [viewpoint_stack[(f*2+1) % scene.maxtime] for f in frame_set]
        else:
            # Pick camera
            method = "random" if iteration < opt.random_until or iteration % 2 == 1 else "by_error"

            cam_no = []
            for _ in range(opt.batch_size):
                last_camera_index = sample_camera(cam_dists, last_camera_index, min_dist)
                cam_no.append(last_camera_index)
            
            viewpoint_cams, sampled_cam_no, sampled_frame_no = image_sampler(method=method, loader=viewpoint_stack, loss_list=loss_list, batch_size=opt.batch_size, \
                cam_no=cam_no, frame_no=sampled_frame_no, total_num_frames=scene.maxtime)
            if iteration >= opt.random_until and opt.num_multiview_ssim > 0 and iteration % 50 < opt.num_multiview_ssim:
                sampled_frame_no = sampled_frame_no  # reuse sampled frame (num_multiview_ssim) times
            else:
                sampled_frame_no = None
        ###
        
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        images = []
        gt_images = []
        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []
        cam_no_list, frame_no_list = [], []
        for viewpoint_cam in viewpoint_cams:
            if type(viewpoint_cam.original_image) == type(None):
                viewpoint_cam.load_image()  # for lazy loading (to avoid OOM issue)
            cam_no = viewpoint_cam.cam_no
            frame_no = viewpoint_cam.frame_no
            cam_no_list.append(cam_no)
            frame_no_list.append(frame_no)
            # print(cam_no, frame_no, viewpoint_cam.image_name)  # for test
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, cam_no=cam_no, iter=iteration, \
                num_down_emb_c=hyper.min_embeddings, num_down_emb_f=hyper.min_embeddings)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            images.append(image.unsqueeze(0))
            gt_image = viewpoint_cam.original_image.cuda()
            gt_images.append(gt_image.unsqueeze(0))
            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)
        
        radii = torch.cat(radii_list,0).max(dim=0).values
        visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
        image_tensor = torch.cat(images,0)
        gt_image_tensor = torch.cat(gt_images,0)

        
        Ll1 = l1_loss(image_tensor, gt_image_tensor, keepdim=True)
        Ll1_items = Ll1.detach()
        Ll1 = Ll1.mean()
        if opt.lambda_dssim > 0. and sampled_frame_no != None or (method == "by_error" and (iteration % 10 == 0) and opt.num_multiview_ssim==0):
            ssim_value, ssim_map = ssim(image_tensor, gt_image_tensor)
            Lssim = (1 - ssim_value) / 2
            loss = Ll1 + opt.lambda_dssim * Lssim
        else:
            loss = Ll1

        psnr_ = psnr(image_tensor, gt_image_tensor).mean().double()
        for i in range(len(Ll1_items)):
            loss_list[cam_no_list[i], frame_no_list[i]] = Ll1_items[i].item()
            # print(i, cam_no_list[i], frame_no_list[i])

        # use l1 instead of opacity reset
        if opt.opacity_l1_coef_fine > 0.:
            loss += opt.opacity_l1_coef_fine * torch.sigmoid(gaussians._opacity.mean())

        # embedding reg using knn (https://github.com/JonathonLuiten/Dynamic3DGaussians)
        if prev_num_pts != gaussians._xyz.shape[0]:
            neighbor_sq_dist, neighbor_indices = o3d_knn(gaussians._xyz.detach().cpu().numpy(), 20)
            neighbor_weight = np.exp(-2000 * neighbor_sq_dist)
            neighbor_indices = torch.tensor(neighbor_indices).cuda().long().contiguous()
            neighbor_weight = torch.tensor(neighbor_weight).cuda().float().contiguous()
            prev_num_pts = gaussians._xyz.shape[0]
        
        emb = gaussians._embedding[:,None,:].repeat(1,20,1)
        emb_knn = gaussians._embedding[neighbor_indices]
        loss += opt.reg_coef * weighted_l2_loss_v2(emb, emb_knn, neighbor_weight)

        # smoothness reg on temporal embeddings
        if opt.coef_tv_temporal_embedding > 0:
            weights = gaussians._deformation.weight
            N, C = weights.shape
            first_difference = weights[1:,:] - weights[N-1,:]
            second_difference = first_difference[1:,:] - first_difference[N-2,:]
            loss += opt.coef_tv_temporal_embedding * torch.square(second_difference).mean()

        
        loss.backward()
        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        for idx in range(0, len(viewspace_point_tensor_list)):
            viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
        iter_end.record()

        if iteration in saving_iterations:
            elapsed_time = time()
            
            total_time_seconds = elapsed_time - start_time
            hours, remainder = divmod(total_time_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            with open(os.path.join(args.model_path, 'training_time.txt'), 'a') as file:
                file.write(f'Iteration {iteration}: {total_time_seconds} seconds ... {int(hours)}h {int(minutes)}m {seconds}sec  points: {gaussians._xyz.shape[0]}\n')

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
            total_point = gaussians._xyz.shape[0]
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}",
                                          "psnr": f"{psnr_:.{2}f}",
                                          "point":f"{total_point}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            timer.pause()
 
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            if dataset.render_process:
                if (iteration < 1000 and iteration % 10 == 1) \
                    or (iteration < 3000 and iteration % 50 == 1) \
                        or (iteration < 10000 and iteration %  100 == 1) \
                            or (iteration < 60000 and iteration % 100 ==1):

                    render_training_image(scene, gaussians, test_cams, render, pipe, background, iteration-1,timer.get_elapsed_time())

            timer.start()
            # Densification
            if iteration < opt.densify_until_iter :
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)
  
                opacity_threshold = opt.opacity_threshold_fine_init - iteration*(opt.opacity_threshold_fine_init - opt.opacity_threshold_fine_after)/(opt.densify_until_iter)  
                densify_threshold = opt.densify_grad_threshold_fine_init - iteration*(opt.densify_grad_threshold_fine_init - opt.densify_grad_threshold_after)/(opt.densify_until_iter )  

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 :
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    
                    gaussians.densify(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
                if iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None

                    gaussians.prune(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)

                    if opt.reset_opacity_ratio > 0 and iteration % opt.pruning_interval == 0:
                        gaussians.reset_opacity(opt.reset_opacity_ratio)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


def training(dataset, hyper, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, expname):
    tb_writer = prepare_output_and_logger(expname)
    gaussians = GaussianModel(dataset.sh_degree, hyper)
    dataset.model_path = args.model_path
    timer = Timer()
    scene = Scene(dataset, gaussians, shuffle=dataset.shuffle, loader=dataset.loader, duration=hyper.total_num_frames, opt=opt)
    timer.start()
    
    start_time = time()
    scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, tb_writer, opt.iterations, timer, start_time)
    end_time = time()
    
    total_time_seconds = end_time - start_time
    hours, remainder = divmod(total_time_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"training time: {int(hours)}h {int(minutes)}m {seconds}sec")


def prepare_output_and_logger(expname):    
    if not args.model_path:
        unique_str = expname

        args.model_path = os.path.join("./output/", unique_str)
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))


        
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
     
     
if __name__ == "__main__":
    # Set up command line argument parser
    # torch.set_default_tensor_type('torch.FloatTensor')
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    setup_seed(6666)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[i*500 for i in range(0,120)])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[3000, 5000, 7000, 14000, 20000, 30000, 45000, 60000, 80000, 100000, 120000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--expname", type=str, default = "")
    parser.add_argument("--configs", type=str, default = "")
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.expname)

    # All done
    print("\nTraining complete.")
