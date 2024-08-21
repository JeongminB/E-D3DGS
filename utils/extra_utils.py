
import torch
import numpy as np
import open3d as o3d

def o3d_knn(pts, num_knn):
    indices = []
    sq_dists = []
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts, np.float64))
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    for p in pcd.points:
        [_, i, d] = pcd_tree.search_knn_vector_3d(p, num_knn + 1)
        indices.append(i[1:])
        sq_dists.append(d[1:])
    return np.array(sq_dists), np.array(indices)


def weighted_l2_loss_v2(x, y, w):
    return torch.sqrt(((x - y) ** 2).sum(-1) * w + 1e-20).mean()


def image_sampler(method="", loader=None, loss_list=None, total_num_frames=50, batch_size=1, cam_no=None, frame_no=None):
    if type(cam_no) == type(None):
        cam_no = np.random.choice(range(len(loader) // total_num_frames), size=batch_size)

    if type(frame_no) == type(None):
        if method == "random":
            frame_no = np.random.choice(range(total_num_frames), size=batch_size)
        elif method == "by_error":
            frame_no = get_idx_by_error(batch_size, loss_list)
    
    # idx = cam_no * total_num_frames + frame_no
    # sampled_image = [loader[idx[0]]]
    # inds = [c * total_num_frames + f for c, f in zip(cam_no, frame_no)]
    sampled_image = [loader[c * total_num_frames + f] for c, f in zip(cam_no, frame_no)]
    return sampled_image, cam_no, frame_no


def get_idx_by_error(batch_size, loss_values):
    loss_values = loss_values.sum(axis=0).reshape(1, -1)
    
    q_low = np.percentile(loss_values, 0)
    q_high = np.percentile(loss_values, 100)
    loss_values = np.clip(loss_values, q_low, q_high)
    loss_values = (loss_values - q_low) / (q_high - q_low)

    normalized_loss = loss_values / np.sum(loss_values)
    probabilities = normalized_loss / normalized_loss.sum()
    cdf = np.cumsum(probabilities, axis=None)
    
    rand_idx = np.random.rand(batch_size)
    idx = np.searchsorted(cdf, rand_idx)
    return idx


def calculate_distances(cameras):
    cameras = np.array(cameras)
    diff = cameras[:, np.newaxis, :] - cameras[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff**2, axis=-1))
    return distances


def sample_camera(distances, last_camera_index, min_distance):
    last_camera_distances = distances[last_camera_index]
    valid_indices = np.where(last_camera_distances >= min_distance)[0]
    valid_indices = valid_indices[valid_indices != last_camera_index]

    if len(valid_indices) > 0:
        return np.random.choice(valid_indices)
    else:
        return np.random.choice(distances.shape[0])