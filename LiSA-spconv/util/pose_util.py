import numpy as np
import pandas as pd
import transforms3d.quaternions as txq
import transforms3d.euler as euler
import struct
import open3d
import torch
import math
import sys
import scipy.interpolate
import scipy.linalg as slin
from scipy.spatial.transform import Rotation as R
from os import path as osp
from data.robotcar_sdk.python.transform import build_se3_transform

def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part last, as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    i, j, k, r = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )

    return o.reshape(quaternions.shape[:-1] + (3, 3))


def vdot(v1, v2):
  """
  Dot product along the dim=1
  :param v1: N x d
  :param v2: N x d
  :return: N x 1
  """
  out = torch.mul(v1, v2)
  out = torch.sum(out, 1)
  return out


def normalize(x, p=2, dim=0):
  """
  Divides a tensor along a certain dim by the Lp norm
  :param x:
  :param p: Lp norm
  :param dim: Dimension to normalize along
  :return:
  """
  xn = x.norm(p=p, dim=dim)
  x = x / xn.unsqueeze(dim=dim)
  return x


def qmult(q1, q2):
  """
  Multiply 2 quaternions
  :param q1: Tensor N x 4
  :param q2: Tensor N x 4
  :return: quaternion product, Tensor N x 4
  """
  q1s, q1v = q1[:, :1], q1[:, 1:]
  q2s, q2v = q2[:, :1], q2[:, 1:]

  qs = q1s*q2s - vdot(q1v, q2v)
  qv = q1v.mul(q2s.expand_as(q1v)) + q2v.mul(q1s.expand_as(q2v)) +\
       torch.cross(q1v, q2v, dim=1)
  q  = torch.cat((qs, qv), dim=1)

  # normalize
  q = normalize(q, dim=1)

  return q


def qinv(q):
  """
  Inverts quaternions
  :param q: N x 4
  :return: q*: N x 4
  """
  q_inv = torch.cat((q[:, :1], -q[:, 1:]), dim=1)
  return q_inv


def qexp_t_safe(q):
  """
  Applies exponential map to log quaternion (safe implementation that does not
  maintain gradient flow)
  :param q: N x 3
  :return: N x 4
  """
  q = torch.from_numpy(np.asarray([qexp(qq) for qq in q.numpy()],
                                  dtype=np.float32))
  return q


def qlog_t_safe(q):
  """
  Applies the log map to a quaternion (safe implementation that does not
  maintain gradient flow)
  :param q: N x 4
  :return: N x 3
  """
  q = torch.from_numpy(np.asarray([qlog(qq) for qq in q.numpy()],
                                  dtype=np.float32))
  return q


def rotate_vec_by_q(t, q):
  """
  rotates vector t by quaternion q
  :param t: vector, Tensor N x 3
  :param q: quaternion, Tensor N x 4
  :return: t rotated by q: t' = t + 2*qs*(qv x t) + 2*qv x (qv x r)
  """
  qs, qv = q[:, :1], q[:, 1:]
  b  = torch.cross(qv, t, dim=1)
  c  = 2 * torch.cross(qv, b, dim=1)
  b  = 2 * b.mul(qs.expand_as(b))
  tq = t + b + c
  return tq


def calc_vo_logq_safe(p0, p1):
  """
  VO in the p0 frame using numpy fns
  :param p0:
  :param p1:
  :return:
  """
  vos_t = p1[:, :3] - p0[:, :3]
  q0 = qexp_t_safe(p0[:, 3:])
  q1 = qexp_t_safe(p1[:, 3:])
  vos_t = rotate_vec_by_q(vos_t, qinv(q0))
  vos_q = qmult(qinv(q0), q1)
  vos_q = qlog_t_safe(vos_q)
  return torch.cat((vos_t, vos_q), dim=1)


def calc_vos_safe_fc(poses):
    """
  calculate the VOs, from a list of consecutive poses (fully connected)
  :param poses: N x T x 7
  :return: N x TC2 x 7
  """
    vos = []
    for p in poses:
        pvos = []
        for i in range(p.size(0)):
            for j in range(i + 1, p.size(0)):
                pvos.append(calc_vo_logq_safe(p[i].unsqueeze(0), p[j].unsqueeze(0)))
        vos.append(torch.cat(pvos, dim=0))
    vos = torch.stack(vos, dim=0)
    return vos


def qlog(q):
    """
    Applies logarithm map to q
    :param q: (4,)
    :return: (3,)
    """
    if all(q[1:] == 0):
        q = np.zeros(3)
    else:
        q = np.arccos(q[0]) * q[1:] / np.linalg.norm(q[1:])

    return q


def qexp(q):
    """
    Applies the exponential map to q
    :param q: (3,)
    :return: (4,)
    """
    n = np.linalg.norm(q)
    q = np.hstack((np.cos(n), np.sinc(n / np.pi) * q))

    return q


def qexp_t(q):
    """
    Applies exponential map to log quaternion
    :param q: N x 3
    :return: N x 4
    """
    n = torch.norm(q, p=2, dim=1, keepdim=True)
    n = torch.clamp(n, min=1e-8)
    q = q * torch.sin(n)
    q = q / n
    q = torch.cat((torch.cos(n), q), dim=1)

    return q


def process_poses(poses_in, mean_t, std_t, align_R, align_t, align_s):
    poses_out = np.zeros((len(poses_in), 6))
    rot_out = np.zeros((len(poses_in), 3, 3))
    poses_out[:, 0:3] = poses_in[:, [3, 7, 11]]

    # align
    for i in range(len(poses_out)):
        R = poses_in[i].reshape((3, 4))[:3, :3]
        rot_out[i, :, :] = R
        q = txq.mat2quat(np.dot(align_R, R))
        q *= np.sign(q[0])  # constrain to hemisphere
        q = qlog(q)
        poses_out[i, 3:] = q
        t = poses_out[i, :3] - align_t
        poses_out[i, :3] = align_s * np.dot(align_R, t[:, np.newaxis]).squeeze()

    # normalize translation
    poses_out[:, :3] -= mean_t
    # poses_out[:, :3] /= std_t

    # max and min
    pose_max = np.max(poses_out[:, :2], axis=0)  # (2,)
    pose_min = np.min(poses_out[:, :2], axis=0)  # (2,)

    return poses_out, rot_out, pose_max, pose_min


def calibrate_process_poses(poses_in, mean_t, align_R, align_t, align_s):
    poses_out = np.zeros((len(poses_in), 6))
    rot_out = np.zeros((len(poses_in), 3, 3))
    poses_out[:, 0:3] = poses_in[:, 9:]

    # align
    for i in range(len(poses_out)):
        R = poses_in[i, :9].reshape((3, 3))
        rot_out[i, :, :] = R
        q = txq.mat2quat(np.dot(align_R, R))
        q *= np.sign(q[0])  # constrain to hemisphere
        q = qlog(q)
        poses_out[i, 3:] = q
        t = poses_out[i, :3] - align_t
        poses_out[i, :3] = align_s * np.dot(align_R, t[:, np.newaxis]).squeeze()

    # normalize translation
    poses_out[:, :3] -= mean_t
    # poses_out[:, :3] /= std_t

    # max and min
    pose_max = np.max(poses_out[:, :2], axis=0)  # (2,)
    pose_min = np.min(poses_out[:, :2], axis=0)  # (2,)

    return poses_out, rot_out, pose_max, pose_min


def val_translation(pred_p, gt_p):
    """
    test model, compute error (numpy)
    input:
        pred_p: [3,]
        gt_p: [3,]
    returns:
        translation error (m):
    """
    if isinstance(pred_p, np.ndarray):
        predicted = pred_p
        groundtruth = gt_p
    else:
        predicted = pred_p.cpu().numpy()
        groundtruth = gt_p.cpu().numpy()
    error = np.linalg.norm(groundtruth - predicted)

    return error


def val_rotation(pred_q, gt_q):
    """
    test model, compute error (numpy)
    input:
        pred_q: [4,]
        gt_q: [4,]
    returns:
        rotation error (degrees):
    """
    if isinstance(pred_q, np.ndarray):
        predicted = pred_q
        groundtruth = gt_q
    else:
        predicted = pred_q.cpu().numpy()
        groundtruth = gt_q.cpu().numpy()

    # d = abs(np.sum(np.multiply(groundtruth, predicted)))
    # if d != d:
    #     print("d is nan")
    #     raise ValueError
    # if d > 1:
    #     d = 1
    # error = 2 * np.arccos(d) * 180 / np.pi0
    # d     = abs(np.dot(groundtruth, predicted))
    # d     = min(1.0, max(-1.0, d))

    d = np.abs(np.dot(groundtruth, predicted))
    d = np.minimum(1.0, np.maximum(-1.0, d))
    error = 2 * np.arccos(d) * 180 / np.pi

    return error


def poses2mats(poses_in):
    poses_out = np.zeros((len(poses_in), 3, 3))  # (B, 3, 3)
    poses_qua = np.asarray([qexp(q) for q in poses_in.cpu().detach().numpy()])

    # align
    for i in range(len(poses_out)):
        R = txq.quat2mat(poses_qua[i])
        poses_out[i, ...] = R

    return poses_out


def lw_poses2mats(poses_in):
    poses_out = np.zeros((len(poses_in), 3, 3))  # (B, 3, 3)
    poses_qua = poses_in

    # align
    for i in range(len(poses_out)):
        R = txq.quat2mat(poses_qua[i])
        poses_out[i, ...] = R

    return poses_out


def estimate_poses(gt_pc, pred_pc, threshold=0.6, device='cuda'):
    # print(source_pc.shape)
    source_pc = gt_pc.cpu().numpy().reshape(-1, 3)
    target_pc = pred_pc[:, :3].cpu().numpy().reshape(-1, 3)
    num_points = source_pc.shape[0]
    pred_t = np.zeros((1, 3))
    pred_q = np.zeros((1, 4))
    index1 = np.arange(0, num_points)
    index2 = np.arange(0, num_points)
    # np.random.shuffle(index1)
    index1 = np.expand_dims(index1, axis=1)
    index2 = np.expand_dims(index2, axis=1)
    corr = np.concatenate((index1, index2), axis=1)

    source_xyz = source_pc
    target_xyz = target_pc
    source = open3d.geometry.PointCloud()
    target = open3d.geometry.PointCloud()
    source.points = open3d.utility.Vector3dVector(source_xyz)
    target.points = open3d.utility.Vector3dVector(target_xyz)
    corres = open3d.utility.Vector2iVector(corr)

    M = open3d.pipelines.registration.registration_ransac_based_on_correspondence(
        source,
        target,
        corres,
        threshold,
        open3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            open3d.pipelines.registration.
            CorrespondenceCheckerBasedOnEdgeLength(0.9),
            open3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(threshold)
        ],
        open3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))

    # refine
    # pred_trans = torch.eye(4)[None].to(device)
    # pred_trans   = torch.from_numpy(M.transformation).to(device, dtype=torch.float32).view(1, 4, 4)
    pred_t[0, :] = M.transformation[:3, 3:].squeeze()
    pred_q[0, :] = txq.mat2quat(M.transformation[:3, :3])
    R            = M.transformation[:3, :3]

    return pred_t, pred_q

def label_estimate_poses(gt_pc, pred_pc, label, threshold=1.4, device='cuda'):
    # print(source_pc.shape)
    # mask = (label == 10) | (label == 12) | (label == 13) | (label == 14)
    mask = (label == 14)
    mask = mask.cpu().numpy().reshape(-1, 1)
    mask = np.tile(mask, (1, 3))
    source_pc = gt_pc.cpu().numpy().reshape(-1, 3)
    target_pc = pred_pc[:, :3].cpu().numpy().reshape(-1, 3)

    source_xyz = source_pc[mask].reshape(-1, 3)
    target_xyz = target_pc[mask].reshape(-1, 3)
    num_points = source_xyz.shape[0]
    pred_t = np.zeros((1, 3))
    pred_q = np.zeros((1, 4))
    index1 = np.arange(0, num_points)
    index2 = np.arange(0, num_points)
    # np.random.shuffle(index1)
    index1 = np.expand_dims(index1, axis=1)
    index2 = np.expand_dims(index2, axis=1)
    corr = np.concatenate((index1, index2), axis=1)

    source = open3d.geometry.PointCloud()
    target = open3d.geometry.PointCloud()
    source.points = open3d.utility.Vector3dVector(source_xyz)
    target.points = open3d.utility.Vector3dVector(target_xyz)
    corres = open3d.utility.Vector2iVector(corr)

    M = open3d.pipelines.registration.registration_ransac_based_on_correspondence(
        source,
        target,
        corres,
        threshold,
        open3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            open3d.pipelines.registration.
                CorrespondenceCheckerBasedOnEdgeLength(0.9),
            open3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(threshold)
        ],
        open3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))

    # pred_trans = torch.eye(4)[None].to(device)
    # pred_trans   = torch.from_numpy(M.transformation).to(device, dtype=torch.float32).view(1, 4, 4)
    pred_t[0, :] = M.transformation[:3, 3:].squeeze()
    pred_q[0, :] = txq.mat2quat(M.transformation[:3, :3])
    R = M.transformation[:3, :3]

    return pred_t, pred_q

def post_refinement(initial_trans, src_keypts, tgt_keypts, weights=None):
    """
    Perform post refinement using the initial transformation matrix, only adopted during testing.
    Input
        - initial_trans: [bs, 4, 4]
        - src_keypts:    [bs, num_corr, 3]
        - tgt_keypts:    [bs, num_corr, 3]
        - weights:       [bs, num_corr]
    Output:
        - final_trans:   [bs, 4, 4]
    """
    assert initial_trans.shape[0] == 1
    inlier_threshold_list = [0.6] * 20
    # print(src_keypts.shape)
    # print(tgt_keypts.shape)
    previous_inlier_num = 0
    for inlier_threshold in inlier_threshold_list:
        warped_src_keypts = transform(src_keypts, initial_trans)
        L2_dis = torch.norm(warped_src_keypts - tgt_keypts, dim=-1)
        pred_inlier = (L2_dis < inlier_threshold)[0]  # assume bs = 1
        inlier_num = torch.sum(pred_inlier)
        if abs(int(inlier_num - previous_inlier_num)) < 1:
            break
        else:
            previous_inlier_num = inlier_num
        initial_trans = rigid_transform_3d(
            A=src_keypts[:, pred_inlier, :],
            B=tgt_keypts[:, pred_inlier, :],
            ## https://link.springer.com/article/10.1007/s10589-014-9643-2
            # weights=None,
            weights=1/(1 + (L2_dis/inlier_threshold)**2)[:, pred_inlier],
            # weights=((1-L2_dis/inlier_threshold)**2)[:, pred_inlier]
        )
    return initial_trans.cpu().numpy().reshape(4, 4)


def transform(pts, trans):
    """
    Applies the SE3 transformations, support torch.Tensor and np.ndarry.  Equation: trans_pts = R @ pts + t
    Input
        - pts: [num_pts, 3] or [bs, num_pts, 3], pts to be transformed
        - trans: [4, 4] or [bs, 4, 4], SE3 transformation matrix
    Output
        - pts: [num_pts, 3] or [bs, num_pts, 3] transformed pts
    """
    if len(pts.shape) == 3:
        trans_pts = trans[:, :3, :3] @ pts.permute(0,2,1) + trans[:, :3, 3:4]
        return trans_pts.permute(0,2,1)
    else:
        trans_pts = trans[:3, :3] @ pts.T + trans[:3, 3:4]
        return trans_pts.T


def rigid_transform_3d(A, B, weights=None, weight_threshold=0):
    """
    Input:
        - A:       [bs, num_corr, 3], source point cloud
        - B:       [bs, num_corr, 3], target point cloud
        - weights: [bs, num_corr]     weight for each correspondence
        - weight_threshold: float,    clips points with weight below threshold
    Output:
        - R, t
    """
    bs = A.shape[0]
    if weights is None:
        weights = torch.ones_like(A[:, :, 0])
    weights[weights < weight_threshold] = 0
    # weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-6)

    # find mean of point cloud
    centroid_A = torch.sum(A * weights[:, :, None], dim=1, keepdim=True) / (torch.sum(weights, dim=1, keepdim=True)[:, :, None] + 1e-6)
    centroid_B = torch.sum(B * weights[:, :, None], dim=1, keepdim=True) / (torch.sum(weights, dim=1, keepdim=True)[:, :, None] + 1e-6)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    # construct weight covariance matrix
    Weight = torch.diag_embed(weights)
    H = Am.permute(0, 2, 1) @ Weight @ Bm

    # find rotation
    U, S, Vt = torch.svd(H.cpu())
    U, S, Vt = U.to(weights.device), S.to(weights.device), Vt.to(weights.device)
    # print(Vt.device)
    # print(Vt.shape)
    # print(U.device)
    # print(U.shape)
    delta_UV = torch.det(Vt @ U.permute(0, 2, 1))
    eye = torch.eye(3)[None, :, :].repeat(bs, 1, 1).to(A.device)
    eye[:, -1, -1] = delta_UV
    R = Vt @ eye @ U.permute(0, 2, 1)
    t = centroid_B.permute(0,2,1) - R @ centroid_A.permute(0,2,1)
    # warp_A = transform(A, integrate_trans(R,t))
    # RMSE = torch.sum( (warp_A - B) ** 2, dim=-1).mean()
    return integrate_trans(R, t)


def integrate_trans(R, t):
    """
    Integrate SE3 transformations from R and t, support torch.Tensor and np.ndarry.
    Input
        - R: [3, 3] or [bs, 3, 3], rotation matrix
        - t: [3, 1] or [bs, 3, 1], translation matrix
    Output
        - trans: [4, 4] or [bs, 4, 4], SE3 transformation matrix
    """
    if len(R.shape) == 3:
        if isinstance(R, torch.Tensor):
            trans = torch.eye(4)[None].repeat(R.shape[0], 1, 1).to(R.device)
        else:
            trans = np.eye(4)[None]
        trans[:, :3, :3] = R
        trans[:, :3, 3:4] = t.view([-1, 3, 1])
    else:
        if isinstance(R, torch.Tensor):
            trans = torch.eye(4).to(R.device)
        else:
            trans = np.eye(4)
        trans[:3, :3] = R
        trans[:3, 3:4] = t
    return trans


def R_t_estimate_poses(source_pc, target_pc, threshold=0.6):
    # print(source_pc.shape)
    num_points = source_pc.shape[0]
    pred_t = np.zeros((1, 3))
    pred_q = np.zeros((1, 4))
    index1 = np.arange(0, num_points)
    index2 = np.arange(0, num_points)
    # np.random.shuffle(index1)
    index1 = np.expand_dims(index1, axis=1)
    index2 = np.expand_dims(index2, axis=1)
    corr = np.concatenate((index1, index2), axis=1)

    source_xyz = source_pc
    target_xyz = target_pc
    source = open3d.geometry.PointCloud()
    target = open3d.geometry.PointCloud()
    source.points = open3d.utility.Vector3dVector(source_xyz)
    target.points = open3d.utility.Vector3dVector(target_xyz)
    corres = open3d.utility.Vector2iVector(corr)

    M = open3d.pipelines.registration.registration_ransac_based_on_correspondence(
        source,
        target,
        corres,
        threshold,
        open3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            open3d.pipelines.registration.
                CorrespondenceCheckerBasedOnEdgeLength(0.9),
            open3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(threshold)
        ],
        open3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))

    pred_t[0, :] = M.transformation[:3, 3:].squeeze()
    pred_q[0, :] = txq.mat2quat(M.transformation[:3, :3])

    return pred_t, pred_q, M.transformation[:3, 3:], M.transformation[:3, :3]


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    Input:
        src: source points, [N, C]
        dst: target points, [M, C]
    Output:
        dist: per-point square distance, [N, M]
    """
    N, _ = src.shape
    M, _ = dst.shape
    dist = -2 * (src @ dst.transpose())
    dist += np.sum(src ** 2, -1).reshape(N, 1)
    dist += np.sum(dst ** 2, -1).reshape(1, M)
    return dist


def plane_estimate_poses(source_pc, target_pc, ground_truth, label, R, t, threshold=0.6):
    plane_index = np.squeeze(label == 1)
    source_plane = source_pc[plane_index, :]
    # 计算转换后的平面
    trans_plane = (R @ source_plane.transpose()).transpose() + t.reshape(1, 3)
    distance = square_distance(trans_plane, target_pc)
    mask = (np.min(distance, 0) <= 15)
    # print(np.sum(mask))
    mask_source_pc = source_pc[mask, :]
    mask_target_pc = target_pc[mask, :]
    mask_ground_truth = ground_truth[mask, :]
    mask_label = label[mask].reshape(-1, 1)
    pred_t, pred_q = estimate_poses(mask_source_pc, mask_target_pc, threshold)

    return pred_t, pred_q, mask_target_pc, mask_ground_truth, mask_label


def only_plane_estimate_poses(source_plane, target_plane, ground_truth, R, t, threshold=0.6):
    """
    It takes in the source and target point clouds, the ground truth, the rotation and translation matrices, and a
    threshold. It then computes the transformed source point cloud, and finds the points in the target point cloud that are
    close to the transformed source point cloud. It then uses the points in the source and target point clouds that are
    close to each other to estimate the pose

    :param source_plane: the source point cloud
    :param target_plane: the plane that we want to estimate the pose of
    :param ground_truth: the ground truth poses
    :param R: rotation matrix
    :param t: translation vector
    :param threshold: the threshold for the distance between the source and target point clouds
    """
    # 计算转换后的平面
    trans_plane = (R @ source_plane.transpose()).transpose() + t.reshape(1, 3)
    distance = square_distance(trans_plane, target_plane)
    mask = (np.min(distance, 0) <= 10)
    # print("*"*20)
    # print(np.sum(mask))
    # print(len(source_plane))
    # print("$" * 20)
    mask_source_pc = source_plane[mask, :]
    mask_target_pc = target_plane[mask, :]
    mask_ground_truth = ground_truth[mask, :]
    pred_t, pred_q = estimate_poses(mask_source_pc, mask_target_pc, threshold)

    return pred_t, pred_q, mask_target_pc, mask_ground_truth


def icp_estimate_pose(source_pc, target_pc, trans_mat_ransac, threshold=0.2):
    # source_pc: 模型输出预测点云
    # targert_pc: 模型输出预测点云*RANSAC求得的变换矩阵
    pred_t = np.zeros((1, 3))
    pred_q = np.zeros((1, 4))
    source_xyz = source_pc
    target_xyz = target_pc
    source = open3d.geometry.PointCloud()
    target = open3d.geometry.PointCloud()
    source.points = open3d.utility.Vector3dVector(source_xyz)
    target.points = open3d.utility.Vector3dVector(target_xyz)
    M = open3d.pipelines.registration.registration_icp(source, target, threshold, np.eye(4),
                                                       open3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                       open3d.pipelines.registration.ICPConvergenceCriteria(
                                                           max_iteration=2000))
    Final_M = M.transformation @ trans_mat_ransac
    pred_t[0, :] = Final_M[:3, 3:].squeeze()
    pred_q[0, :] = txq.mat2quat(Final_M[:3, :3])

    return pred_t, pred_q


def ds_pc(cloud, target_num):
    if cloud.shape[0] < target_num:
        # Add in artificial points if necessary
        # print('Only %i out of %i required points in raw point cloud. Duplicating...' % (cloud.shape[0], target_num))
        num_to_pad = target_num - cloud.shape[0]
        index = np.random.choice(cloud.shape[0], size=num_to_pad, replace=True)
        pad_points = cloud[index, :]
        cloud = np.concatenate((cloud, pad_points), axis=0)

        return cloud
    else:
        index = np.random.choice(cloud.shape[0], size=target_num, replace=True)
        cloud = cloud[index, :]

        return cloud


def filter_overflow_ts(filename, ts_raw):
    #
    file_data = pd.read_csv(filename)
    base_name = osp.basename(filename)

    if base_name.find('vo') > -1:
        ts_key = 'source_timestamp'
    else:
        ts_key = 'timestamp'

    pose_timestamps = file_data[ts_key].values
    min_pose_timestamps = min(pose_timestamps)
    max_pose_timestamps = max(pose_timestamps)
    ts_filted = [t for t in ts_raw if min_pose_timestamps < t < max_pose_timestamps]
    abandon_num = len(ts_raw) - len(ts_filted)
    print('abandom %d pointclouds that exceed the range of %s' % (abandon_num, filename))

    return ts_filted


def filter_overflow_nclt(gt_filename, ts_raw):  # 滤波函数
    # gt_filename: GT对应的文件名
    # ts_raw: 原始数据集提供的点云时间戳
    ground_truth = np.loadtxt(gt_filename, delimiter=",")[1:, 0]
    min_pose_timestamps = min(ground_truth)
    max_pose_timestamps = max(ground_truth)
    ts_filted = [t for t in ts_raw if min_pose_timestamps < t < max_pose_timestamps]
    abandon_num = len(ts_raw) - len(ts_filted)
    print('abandom %d pointclouds that exceed the range of %s' % (abandon_num, gt_filename))

    return ts_filted


def interpolate_pose_nclt(gt_filename, ts_raw):  # 插值函数
    # gt_filename: GT对应文件名
    # ts_raw: 滤波后的点云时间戳
    ground_truth = np.loadtxt(gt_filename, delimiter=",")
    ground_truth = ground_truth[np.logical_not(np.any(np.isnan(ground_truth), 1))]
    interp = scipy.interpolate.interp1d(ground_truth[:, 0], ground_truth[:, 1:], kind='nearest', axis=0)
    pose_gt = interp(ts_raw)

    return pose_gt


def so3_to_euler_nclt(poses_in):
    N = len(poses_in)
    poses_out = np.zeros((N, 4, 4))
    for i in range(N):
        poses_out[i, :, :] = build_se3_transform([poses_in[i, 0], poses_in[i, 1], poses_in[i, 2],
                                                  poses_in[i, 3], poses_in[i, 4], poses_in[i, 5]])

    return poses_out


def convert_nclt(x_s, y_s, z_s):  # 输入点云转换函数
    # 文档种提供的转换函数
    # 原文档返回为x, y, z，但在绘制可视化图时z取负，此处先取负
    scaling = 0.005  # 5 mm
    offset = -100.0

    x = x_s * scaling + offset
    y = y_s * scaling + offset
    z = z_s * scaling + offset

    return x, y, -z


def load_velodyne_binary_nclt(filename):  # 读入二进制点云
    f_bin = open(filename, "rb")
    hits = []
    while True:
        x_str = f_bin.read(2)
        if x_str == b'':  # eof
            break
        x = struct.unpack('<H', x_str)[0]
        y = struct.unpack('<H', f_bin.read(2))[0]
        z = struct.unpack('<H', f_bin.read(2))[0]
        i = struct.unpack('B', f_bin.read(1))[0]
        l = struct.unpack('B', f_bin.read(1))[0]

        x, y, z = convert_nclt(x, y, z)

        hits += [[x, y, z]]

    f_bin.close()

    hits = np.array(hits)

    return hits

def interpolate_pose_xmu(gt_filename, ts_raw):  # 插值函数
    # gt_filename: GT对应文件名
    # ts_raw: 滤波后的点云时间戳
    ground_truth = np.loadtxt(gt_filename)
    ground_truth = ground_truth[np.logical_not(np.any(np.isnan(ground_truth), 1))]
    interp = scipy.interpolate.interp1d(ground_truth[:, 0], ground_truth[:, 1:], kind='nearest', axis=0)
    pose_gt = interp(ts_raw)

    return pose_gt

def filter_overflow_xmu(filename, ts_raw):  # 滤波函数
    #
    file_data = np.loadtxt(filename)
    # base_name = osp.basename(filename)
    # mask = file_data[:, -1] == 2
    # pose_timestamps = file_data[mask, 0]
    pose_timestamps = file_data[:, 0]
    min_pose_timestamps = min(pose_timestamps)
    max_pose_timestamps = max(pose_timestamps)
    ts_filted = [t for t in ts_raw if min_pose_timestamps < t < max_pose_timestamps]
    abandon_num = len(ts_raw) - len(ts_filted)
    print('abandom %d pointclouds that exceed the range of %s' % (abandon_num, filename))

    return ts_filted

def knn(x, k, ignore_self=False, normalized=True):
    """ find feature space knn neighbor of x
    Input:
        - x:       [bs, num_corr, num_channels],  input features
        - k:
        - ignore_self:  True/False, return knn include self or not.
        - normalized:   True/False, if the feature x normalized.
    Output:
        - idx:     [bs, num_corr, k], the indices of knn neighbors
    """
    inner = 2 * torch.matmul(x, x.transpose(2, 1))
    if normalized:
        pairwise_distance = 2 - inner
    else:
        xx = torch.sum(x ** 2, dim=-1, keepdim=True)
        pairwise_distance = xx - inner + xx.transpose(2, 1)

    if ignore_self is False:
        idx = pairwise_distance.topk(k=k, dim=-1, largest=False)[1]  # (batch_size, num_points, k)
    else:
        idx = pairwise_distance.topk(k=k + 1, dim=-1, largest=False)[1][:, :, 1:]
    return idx


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    # batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    # new_points = points[batch_indices, idx, :]
    new_points = torch.squeeze(points[idx, :])
    return new_points


def skew(x):
    """
  returns skew symmetric matrix from vector
  :param x: 3 x 1
  :return:
  """
    s = np.asarray([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])
    return s


def dpq_q(p):
    """
  returns the jacobian of quaternion product pq w.r.t. q
  :param p: 4 x 1
  :return: 4 x 4
  """
    J = np.zeros((4, 4))
    J[0, 0] = p[0]
    J[0, 1:] = -p[1:].squeeze()
    J[1:, 0] = p[1:].squeeze()
    J[1:, 1:] = p[0] * np.eye(3) + skew(p[1:])
    return J


def dpsq_q(p):
    """
  returns the jacobian of quaternion product (p*)q w.r.t. q
  :param p: 4 x 1
  :return: 4 x 4
  """
    J = np.zeros((4, 4))
    J[0, 0] = p[0]
    J[0, 1:] = -p[1:].squeeze()
    J[1:, 0] = -p[1:].squeeze()
    J[1:, 1:] = p[0] * np.eye(3) - skew(p[1:])
    return J


def dpsq_p(q):
    """
  returns the jacobian of quaternion product (p*)q w.r.t. p
  :param q: 4 x 1
  :return: 4 x 4
  """
    J = np.zeros((4, 4))
    J[0, 0] = q[0]
    J[0, 1:] = q[1:].squeeze()
    J[1:, 0] = q[1:].squeeze()
    J[1:, 1:] = -q[0] * np.eye(3) + skew(q[1:])
    return J


def dqstq_q(q, t):
    """
  jacobian of q* t q w.r.t. q
  :param q: 4 x 1
  :param t: 3 x 1
  :return: 3 x 4
  """
    J = np.zeros((3, 4))
    J[:, :1] = q[0] * t - np.cross(q[1:], t, axis=0)
    J[:, 1:] = -np.dot(t, q[1:].T) + np.dot(t.T, q[1:]) * np.eye(3) + \
               np.dot(q[1:], t.T) + q[0] * skew(t)
    J *= 2
    return J


def dqstq_t(q):
    """
  jacobian of q* t q w.r.t. t
  :param q: 4 x 1
  :return: 3 x 3
  """
    J = (q[0] * q[0] - np.dot(q[1:].T, q[1:])) * np.eye(3) + 2 * np.dot(q[1:], q[1:].T) - \
        2 * q[0] * skew(q[1:])
    return J


def m_rot(x):
    """
  returns Jacobian of exponential map w.r.t. manifold increment
  :param x: part of state vector affected by increment, 4 x 1
  :return: 4 x 3
  """
    # jacobian of full q wrt qm (quaternion update on manifold),
    # evaluated at qv = (0, 0, 0)
    # full q is derived using either the exponential map or q0 = sqrt(1-qm^2)
    jm = np.vstack((np.zeros((1, 3)), np.eye(3)))  # 4 x 3
    m = np.dot(dpq_q(p=x), jm)
    return m


class PoseGraph:
    def __init__(self):
        """
    implements pose graph optimization from
    "Hybrid Hessians for Optimization of Pose Graphs" - Y. LeCun et al
    and "A Tutorial on Graph-Based SLAM" - W. Burgard et al
    """
        self.N = 0
        self.z = np.zeros((0, 0))

    def jacobian(self, L_ax, L_aq, L_rx, L_rq):
        J = np.zeros((0, 6 * self.N))  # 6 because updates for rotation are on manifold

        # unary constraints
        for i in range(self.N):
            # translation constraint
            jt = np.zeros((3, J.shape[1]))
            jt[:, 6 * i: 6 * i + 3] = np.eye(3)
            J = np.vstack((J, np.dot(L_ax, jt)))

            # rotation constraint
            jr = np.zeros((4, J.shape[1]))
            jr[:, 6 * i + 3: 6 * i + 6] = m_rot(x=self.z[7 * i + 3: 7 * i + 7])
            J = np.vstack((J, np.dot(L_aq, jr)))

        # pairwise constraints
        for i in range(self.N - 1):
            # translation constraint
            jt = np.zeros((3, J.shape[1]))
            dt = dqstq_t(q=self.z[7 * i + 3: 7 * i + 7])
            # dt = np.eye(3)
            jt[:, 6 * i: 6 * i + 3] = -dt
            jt[:, 6 * (i + 1): 6 * (i + 1) + 3] = dt
            # m = m_rot(x=self.z[7*i+3 : 7*i+7])
            # a = dqstq_q(q=self.z[7*i+3 : 7*i+7],
            #             t=self.z[7*(i+1) : 7*(i+1)+3]-self.z[7*i : 7*i+3])
            # jt[:, 6*i+3 : 6*i+6] = np.dot(a, m)
            J = np.vstack((J, np.dot(L_rx, jt)))

            # rotation constraint
            jr = np.zeros((4, J.shape[1]))
            m = m_rot(x=self.z[7 * i + 3: 7 * i + 7])
            a = dpsq_p(q=self.z[7 * (i + 1) + 3: 7 * (i + 1) + 7])
            jr[:, 6 * i + 3: 6 * i + 6] = np.dot(a, m)
            m = m_rot(x=self.z[7 * (i + 1) + 3: 7 * (i + 1) + 7])
            b = dpsq_q(p=self.z[7 * i + 3: 7 * i + 7])
            jr[:, 6 * (i + 1) + 3: 6 * (i + 1) + 6] = np.dot(b, m)
            J = np.vstack((J, np.dot(L_rq, jr)))

        return J

    def residuals(self, poses, vos, L_ax, L_aq, L_rx, L_rq):
        """
    computes the residuals
    :param poses: N x 7
    :param vos: (N-1) x 7
    :param L_ax: 3 x 3
    :param L_aq: 4 x 4
    :param L_rx: 3 x 3
    :param L_rq: 4 x 4
    :return:
    """
        r = np.zeros((0, 1))

        # unary residuals
        L = np.zeros((7, 7))
        L[:3, :3] = L_ax
        L[3:, 3:] = L_aq
        for i in range(self.N):
            rr = self.z[7 * i: 7 * (i + 1)] - np.reshape(poses[i], (-1, 1))
            r = np.vstack((r, np.dot(L, rr)))

        # pairwise residuals
        for i in range(self.N - 1):
            # translation residual
            v = self.z[7 * (i + 1):7 * (i + 1) + 3, 0] - self.z[7 * i:7 * i + 3, 0]
            q = txq.qinverse(self.z[7 * i + 3:7 * i + 7, 0])
            rt = txq.rotate_vector(v, q)
            rt = rt[:, np.newaxis] - vos[i, :3].reshape((-1, 1))
            # rt = self.z[7*(i+1) : 7*(i+1)+3] - self.z[7*i : 7*i+3] - \
            #     vos[i, :3].reshape((-1, 1))
            r = np.vstack((r, np.dot(L_rx, rt)))

            # rotation residual
            q0 = self.z[7 * i + 3: 7 * i + 7].squeeze()
            q1 = self.z[7 * (i + 1) + 3: 7 * (i + 1) + 7].squeeze()
            qvo = txq.qmult(txq.qinverse(q0), q1).reshape((-1, 1))
            rq = qvo - vos[i, 3:].reshape((-1, 1))
            r = np.vstack((r, np.dot(L_rq, rq)))

        return r

    def update_on_manifold(self, x):
        """
    Updates the state vector on manifold
    :param x: manifold increment, column vector
    :return:
    """
        for i in range(self.N):
            # update translation
            t = x[6 * i: 6 * i + 3]
            self.z[7 * i: 7 * i + 3] += t

            # update rotation
            qm = x[6 * i + 3: 6 * i + 6]  # quaternion on the manifold
            dq = np.zeros(4)
            # method in Burgard paper
            # dq[1:] = qm.squeeze()
            # dq[0] = math.sqrt(1 - sum(np.square(qm)))  # incremental quaternion
            # method of exponential map
            n = np.linalg.norm(qm)
            dq[0] = math.cos(n)
            dq[1:] = np.sinc(n / np.pi) * qm.squeeze()
            q = self.z[7 * i + 3: 7 * i + 7].squeeze()
            q = txq.qmult(q, dq).reshape((-1, 1))
            self.z[7 * i + 3: 7 * i + 7] = q

    def optimize(self, poses, vos, sax=1, saq=1, srx=1, srq=1, n_iters=10):
        """
    run PGO, with init = poses
    :param poses:
    :param vos:
    :param sax: sigma for absolute translation
    :param saq: sigma for absolute rotation
    :param srx: sigma for relative translation
    :param srq: sigma for relative rotation
    :param n_iters:
    :return:
    """
        self.N = len(poses)
        # init state vector with the predicted poses
        self.z = np.reshape(poses.copy(), (-1, 1))

        # construct the information matrices
        L_ax = np.linalg.cholesky(np.eye(3) / sax)
        L_aq = np.linalg.cholesky(np.eye(4) / saq)
        L_rx = np.linalg.cholesky(np.eye(3) / srx)
        L_rq = np.linalg.cholesky(np.eye(4) / srq)

        for n_iter in range(n_iters):
            J = self.jacobian(L_ax.T, L_aq.T, L_rx.T, L_rq.T)
            r = self.residuals(poses.copy(), vos.copy(), L_ax.T, L_aq.T, L_rx.T,
                               L_rq.T)
            H = np.dot(J.T, J)  # hessian
            b = np.dot(J.T, r)  # residuals

            # solve Hx = -b for x
            R = slin.cholesky(H)  # H = R' R
            y = slin.solve_triangular(R.T, -b)
            x = slin.solve_triangular(R, y)

            self.update_on_manifold(x)

        return self.z.reshape((-1, 7))


class PoseGraphFC:
    def __init__(self):
        """
    implements pose graph optimization from
    "Hybrid Hessians for Optimization of Pose Graphs" - Y. LeCun et al
    and "A Tutorial on Graph-Based SLAM" - W. Burgard et al
    fully connected version
    """
        self.N = 0
        self.z = np.zeros((0, 0))

    def jacobian(self, L_ax, L_aq, L_rx, L_rq):
        J = np.zeros((0, 6 * self.N))  # 6 because updates for rotation are on manifold

        # unary constraints
        for i in range(self.N):
            # translation constraint
            jt = np.zeros((3, J.shape[1]))
            jt[:, 6 * i: 6 * i + 3] = np.eye(3)
            J = np.vstack((J, np.dot(L_ax, jt)))

            # rotation constraint
            jr = np.zeros((4, J.shape[1]))
            jr[:, 6 * i + 3: 6 * i + 6] = m_rot(x=self.z[7 * i + 3: 7 * i + 7])
            J = np.vstack((J, np.dot(L_aq, jr)))

        # pairwise constraints
        for i in range(self.N):
            for j in range(i + 1, self.N):
                # translation constraint
                jt = np.zeros((3, J.shape[1]))
                dt = dqstq_t(q=self.z[7 * i + 3: 7 * i + 7])
                # dt = np.eye(3)
                jt[:, 6 * i: 6 * i + 3] = -dt
                jt[:, 6 * j: 6 * j + 3] = dt
                # m = m_rot(x=self.z[7*i+3 : 7*i+7])
                # a = dqstq_q(q=self.z[7*i+3 : 7*i+7],
                #             t=self.z[7*(i+1) : 7*(i+1)+3]-self.z[7*i : 7*i+3])
                # jt[:, 6*i+3 : 6*i+6] = np.dot(a, m)
                J = np.vstack((J, np.dot(L_rx, jt)))

                # rotation constraint
                jr = np.zeros((4, J.shape[1]))
                m = m_rot(x=self.z[7 * i + 3: 7 * i + 7])
                a = dpsq_p(q=self.z[7 * j + 3: 7 * j + 7])
                jr[:, 6 * i + 3: 6 * i + 6] = np.dot(a, m)
                m = m_rot(x=self.z[7 * j + 3: 7 * j + 7])
                b = dpsq_q(p=self.z[7 * i + 3: 7 * i + 7])
                jr[:, 6 * j + 3: 6 * j + 6] = np.dot(b, m)
                J = np.vstack((J, np.dot(L_rq, jr)))

        return J

    def residuals(self, poses, vos, L_ax, L_aq, L_rx, L_rq):
        """
    computes the residuals
    :param poses: N x 7
    :param vos: (N-1) x 7
    :param L_ax: 3 x 3
    :param L_aq: 4 x 4
    :param L_rx: 3 x 3
    :param L_rq: 4 x 4
    :return:
    """
        r = np.zeros((0, 1))

        # unary residuals
        L = np.zeros((7, 7))
        L[:3, :3] = L_ax
        L[3:, 3:] = L_aq
        for i in range(self.N):
            rr = self.z[7 * i: 7 * (i + 1)] - np.reshape(poses[i], (-1, 1))
            r = np.vstack((r, np.dot(L, rr)))

        # pairwise residuals
        k = 0
        for i in range(self.N):
            for j in range(i + 1, self.N):
                # translation residual
                v = self.z[7 * j:7 * j + 3, 0] - self.z[7 * i:7 * i + 3, 0]
                q = txq.qinverse(self.z[7 * i + 3:7 * i + 7, 0])
                rt = txq.rotate_vector(v, q)
                rt = rt[:, np.newaxis] - vos[k, :3].reshape((-1, 1))
                # rt = self.z[7*(i+1) : 7*(i+1)+3] - self.z[7*i : 7*i+3] - \
                #     vos[i, :3].reshape((-1, 1))
                r = np.vstack((r, np.dot(L_rx, rt)))

                # rotation residual
                q0 = self.z[7 * i + 3: 7 * i + 7].squeeze()
                q1 = self.z[7 * j + 3: 7 * j + 7].squeeze()
                qvo = txq.qmult(txq.qinverse(q0), q1).reshape((-1, 1))
                rq = qvo - vos[k, 3:].reshape((-1, 1))
                r = np.vstack((r, np.dot(L_rq, rq)))
                k += 1

        return r

    def update_on_manifold(self, x):
        """
    Updates the state vector on manifold
    :param x: manifold increment, column vector
    :return:
    """
        for i in range(self.N):
            # update translation
            t = x[6 * i: 6 * i + 3]
            self.z[7 * i: 7 * i + 3] += t

            # update rotation
            qm = x[6 * i + 3: 6 * i + 6]  # quaternion on the manifold
            dq = np.zeros(4)
            # method in Burgard paper
            # dq[1:] = qm.squeeze()
            # dq[0] = math.sqrt(1 - sum(np.square(qm)))  # incremental quaternion
            # method of exponential map
            n = np.linalg.norm(qm)
            dq[0] = math.cos(n)
            dq[1:] = np.sinc(n / np.pi) * qm.squeeze()
            q = self.z[7 * i + 3: 7 * i + 7].squeeze()
            q = txq.qmult(q, dq).reshape((-1, 1))
            self.z[7 * i + 3: 7 * i + 7] = q

    def optimize(self, poses, vos, sax=1, saq=1, srx=1, srq=1, n_iters=10):
        """
    run PGO, with init = poses
    :param poses:
    :param vos:
    :param sax: sigma for absolute translation
    :param saq: sigma for absolute rotation
    :param srx: sigma for relative translation
    :param srq: sigma for relative rotation
    :param n_iters:
    :return:
    """
        self.N = len(poses)
        # init state vector with the predicted poses
        self.z = np.reshape(poses.copy(), (-1, 1))

        # construct the information matrices
        L_ax = np.linalg.cholesky(np.eye(3) / sax)
        L_aq = np.linalg.cholesky(np.eye(4) / saq)
        L_rx = np.linalg.cholesky(np.eye(3) / srx)
        L_rq = np.linalg.cholesky(np.eye(4) / srq)

        for n_iter in range(n_iters):
            J = self.jacobian(L_ax.T, L_aq.T, L_rx.T, L_rq.T)
            r = self.residuals(poses.copy(), vos.copy(), L_ax.T, L_aq.T, L_rx.T,
                               L_rq.T)
            H = np.dot(J.T, J)  # hessian
            b = np.dot(J.T, r)  # residuals

            # solve Hx = -b for x
            R = slin.cholesky(H)  # H = R' R
            y = slin.solve_triangular(R.T, -b)
            x = slin.solve_triangular(R, y)

            self.update_on_manifold(x)

        return self.z.reshape((-1, 7))


def optimize_poses(pred_poses, vos=None, fc_vos=False, target_poses=None,
                   sax=1, saq=1, srx=20, srq=20):
    """
  optimizes poses using either the VOs or the target poses (calculates VOs
  from them)
  :param pred_poses: N x 7
  :param vos: (N-1) x 7
  :param fc_vos: whether to use relative transforms between all frames in a fully
  connected manner, not just consecutive frames
  :param target_poses: N x 7
  :param: sax: covariance of pose translation (1 number)
  :param: saq: covariance of pose rotation (1 number)
  :param: srx: covariance of VO translation (1 number)
  :param: srq: covariance of VO rotation (1 number)
  :return:
  """
    pgo = PoseGraphFC() if fc_vos else PoseGraph()
    if vos is None:
        if target_poses is not None:
            # calculate the VOs (in the pred_poses frame)
            vos = np.zeros((len(target_poses) - 1, 7))
            for i in range(len(vos)):
                vos[i, :3] = target_poses[i + 1, :3] - target_poses[i, :3]
                q0 = target_poses[i, 3:]
                q1 = target_poses[i + 1, 3:]
                vos[i, 3:] = txq.qmult(txq.qinverse(q0), q1)
        else:
            print('Specify either VO or target poses')
            return None
    optim_poses = pgo.optimize(poses=pred_poses, vos=vos, sax=sax, saq=saq,
                               srx=srx, srq=srq)
    return optim_poses
