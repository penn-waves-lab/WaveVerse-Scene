import torch
from smplx.lbs import batch_rodrigues
from . import config


def temporal_rotation_loss(body_pose, global_orient, weight):
    """Discourage frame-to-frame twists without penalizing axis-angle wrapping."""
    if weight == 0 or len(body_pose) < 2:
        return body_pose.sum() * 0
    pose = torch.cat([global_orient, body_pose], dim=-1)
    rotations = batch_rodrigues(pose.reshape(-1, 3)).reshape(len(pose), -1, 3, 3)
    return weight ** 2 * (rotations[1:] - rotations[:-1]).square().sum()


def floor_penetration_loss(vertices, translation, floor_height, weight=600.0):
    """Penalize fitted skin below a known Y-up floor, rather than joint centers."""
    if floor_height is None:
        return vertices.new_zeros(())
    height = vertices[..., 1] + translation[..., 1]
    return weight ** 2 * torch.relu(floor_height - height).square().sum()

def gmof(x, sigma):
    """
    Geman-McClure error function
    """
    x_squared = x ** 2
    sigma_squared = sigma ** 2
    return sigma_squared * x_squared / (sigma_squared + x_squared)

def angle_prior(pose):
    """
    Angle prior that penalizes unnatural bending of the knees and elbows
    """
    return torch.exp(pose[:, [55 - 3, 58 - 3, 12 - 3, 15 - 3]] * torch.tensor([1.0, -1.0, -1, -1.0], device=pose.device)) ** 2

def body_fitting_loss_3d(body_pose, preserve_pose, betas, model_joints, camera_translation, j3d, pose_prior, joints3d_conf, sigma=100, pose_prior_weight=4.78 * 1.5, shape_prior_weight=5.0, angle_prior_weight=15.2, joint_loss_weight=500.0, pose_preserve_weight=0.0, use_collision=False, model_vertices=None, model_faces=None, search_tree=None, pen_distance=None, filter_faces=None, collision_loss_weight=1000):
    """
    Loss function for body fitting
    """
    batch_size = body_pose.shape[0]
    joint3d_error = gmof(model_joints + camera_translation - j3d, sigma)
    joint3d_loss_part = joints3d_conf ** 2 * joint3d_error.sum(dim=-1)
    joint3d_loss = (joint_loss_weight ** 2 * joint3d_loss_part).sum(dim=-1)
    pose_prior_loss = pose_prior_weight ** 2 * pose_prior(body_pose, betas)
    angle_prior_loss = angle_prior_weight ** 2 * angle_prior(body_pose).sum(dim=-1)
    shape_prior_loss = shape_prior_weight ** 2 * (betas ** 2).sum(dim=-1)
    collision_loss = 0.0
    if use_collision:
        triangles = torch.index_select(model_vertices, 1, model_faces).view(batch_size, -1, 3, 3)
        with torch.no_grad():
            collision_idxs = search_tree(triangles)
        if filter_faces is not None:
            collision_idxs = filter_faces(collision_idxs)
        if collision_idxs.ge(0).sum().item() > 0:
            collision_loss = torch.sum(collision_loss_weight * pen_distance(triangles, collision_idxs))
    pose_preserve_loss = pose_preserve_weight ** 2 * ((body_pose - preserve_pose) ** 2).sum(dim=-1)
    total_loss = joint3d_loss + pose_prior_loss + angle_prior_loss + shape_prior_loss + collision_loss + pose_preserve_loss
    return total_loss.sum()

def camera_fitting_loss_3d(model_joints, camera_t, camera_t_est, j3d, joints_category='orig', depth_loss_weight=100.0):
    """
    Loss function for camera optimization.
    """
    model_joints = model_joints + camera_t
    gt_joints = ['RHip', 'LHip', 'RShoulder', 'LShoulder']
    gt_joints_ind = [config.JOINT_MAP[joint] for joint in gt_joints]
    if joints_category == 'orig':
        select_joints_ind = [config.JOINT_MAP[joint] for joint in gt_joints]
    elif joints_category == 'AMASS':
        select_joints_ind = [config.AMASS_JOINT_MAP[joint] for joint in gt_joints]
    else:
        print('NO SUCH JOINTS CATEGORY!')
    j3d_error_loss = (j3d[:, select_joints_ind] - model_joints[:, gt_joints_ind]) ** 2
    depth_loss = depth_loss_weight ** 2 * (camera_t - camera_t_est) ** 2
    total_loss = j3d_error_loss + depth_loss
    return total_loss.sum()
