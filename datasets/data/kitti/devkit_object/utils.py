
import numpy as np


def get_dataset_classes():
    return ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc']


def name_2_label(name):
    names = get_dataset_classes()
    if isinstance(name, (list, tuple, np.ndarray)):
        label = [name_2_label(n) for n in name]
        if isinstance(name, np.ndarray):
            label = np.array(label)
        return label
    elif isinstance(name, str):
        label = names.index(name) if names.__contains__(name) else -1
        return label
    else:
        raise Exception('name must be in (str, list, tuple, np.ndarray)')


def box_iou(box1, box2):
    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    high_xy = np.minimum(box1[:, 2:], box2[:, 2:])
    low_xy = np.maximum(box1[:, :2], box2[:, :2])

    inter = np.clip((high_xy - low_xy), a_min=0, a_max=1280).prod(-1)

    return inter / (area1 + area2 - inter)


def calc_bbox_max_and_min_area(path, objs=None):
    if objs is None:
        objs = get_dataset_classes()
    cls = [name_2_label(obj) for obj in objs]
    labels = np.load(path, allow_pickle=True)
    labels = np.concatenate(labels, axis=0)
    keep = np.zeros_like(labels[:, 0])
    for i in cls:
        keep[labels[:, 0] == int(i)] = 1
    keep = keep.astype(np.bool)
    labels = labels[keep]
    bboxes = labels[:, 1:5]
    w = (bboxes[:, 2] - bboxes[:, 0])
    h = (bboxes[:, 3] - bboxes[:, 1])
    keep = (w >= 2) & (h >= 2)
    w = w[keep]
    h = h[keep]
    areas = w * h
    max_area = np.amax(areas)
    min_area = np.amin(areas)
    return max_area, min_area


def rotation_matrix(yaw, pitch=0, roll=0):
    N = len(yaw)
    tx = roll
    ty = yaw
    tz = pitch
    sin_tx = np.sin(tx)
    cos_tx = np.cos(tx)
    sin_ty = np.sin(ty)
    cos_ty = np.cos(ty)
    sin_tz = np.sin(tz)
    cos_tz = np.cos(tz)
    Rx, Ry, Rz = np.zeros((N, 9)), np.zeros((N, 9)), np.zeros((N, 9))
    Rx[:, 0] = 1
    Rx[:, 4] = cos_tx
    Rx[:, 5] = -sin_tx
    Rx[:, 7] = sin_tx
    Rx[:, 8] = cos_tx

    Ry[:, 0] = cos_ty
    Ry[:, 2] = sin_ty
    Ry[:, 4] = 1
    Ry[:, 6] = -sin_ty
    Ry[:, 8] = cos_ty

    Rz[:, 0] = cos_tz
    Rz[:, 1] = -sin_tz
    Rz[:, 3] = sin_tz
    Rz[:, 4] = cos_tz
    Rz[:, 8] = 1

    # Rx = np.array([1, 0, 0, 0, cos_tx, -sin_tx, 0, sin_tx, cos_tx])
    # Ry = np.array([cos_ty, 0, sin_ty, 0, 1, 0, -sin_ty, 0, cos_ty])
    # Rz = np.array([cos_tz, -sin_tz, 0, sin_tz, cos_tz, 0, 0, 0, 1])

    return Ry.reshape([N, 3, 3])


def calc_proj2d_bbox3d(dim, loc, ry, K, xrange=(-200, 1500), yrange=(-200, 500)):
    # dim (h, w, l)
    dim = np.vstack([dim[:, 2], dim[:, 0], dim[:, 1]]).T
    N = len(ry)
    R = rotation_matrix(ry)
    dim_offset = np.expand_dims(np.eye(3) / 2, axis=0).repeat([N], axis=0)
    dM = dim_offset * dim.reshape(N, 1, 3)
    M = np.matmul(R, dM)
    x_corners = []
    y_corners = []
    z_corners = []
    #             / x
    #            /
    #    z -----
    #           |
    #           | y
    #          2----------3
    #         /|         /|
    #        / |        / |
    #       /  0-------/--1
    #      /  /       /  /
    #     6--/-------7  /
    #     | /        | /
    #     |/         |/
    #     4----------5
    for i in [1, -1]:  # x
        for j in [1, -1]:  # y
            for k in [1, -1]:  # z
                x_corners.append(i)
                y_corners.append(j)
                z_corners.append(k)
    x_corners.append(0)
    y_corners.append(0)
    z_corners.append(0)
    corners = np.vstack([x_corners, y_corners, z_corners])
    corners = np.expand_dims(corners, axis=0).repeat([N], axis=0)

    corners = np.matmul(M, corners)
    location = loc.reshape(N, 3, 1)
    corners += location
    # mask = np.amin(corners[:, 2, :], axis=-1) > 1e-4
    proj_2d = np.matmul(K, corners)
    proj_2d[:, :2, :] /= (proj_2d[:, None, 2, :] + 1e-10)
    proj_2d = proj_2d[:, :2, :]

    oc_min = np.amin(proj_2d, axis=-1)
    oc_max = np.amax(proj_2d, axis=-1)

    mask = (oc_min[:, 0] > xrange[0]) & (oc_max[:, 0] < xrange[1]) & \
           (oc_min[:, 1] > yrange[0]) & (oc_max[:, 1] < yrange[1])
    return proj_2d, np.concatenate([oc_min, oc_max], axis=-1), mask


def calc_offset_vertex_to_center(path, k_path, shape_path, objs=None):
    if objs is None:
        objs = get_dataset_classes()
    cls = [name_2_label(obj) for obj in objs]
    labels = np.load(path, allow_pickle=True)
    K = np.load(k_path, allow_pickle=True)
    shapes = np.load(shape_path, allow_pickle=True)
    NK = []
    nshapes = []
    for i, l, k, s in zip(range(len(labels)), labels, K, shapes):
        NK.append(k.reshape(1, 3, 3).repeat([len(l)], axis=0))
        nshapes.append(s.reshape(1, 2).repeat([len(l)], axis=0))
    labels = np.concatenate(labels, axis=0)
    K = np.concatenate(NK, axis=0)
    shapes = np.concatenate(nshapes, axis=0)
    keep = np.zeros_like(labels[:, 0])
    for i in cls:
        keep[labels[:, 0] == int(i)] = 1
    keep = keep.astype(np.bool)
    labels = labels[keep]
    K = K[keep]
    shapes = shapes[keep]
    bboxes = labels[:, 1:5]
    w = (bboxes[:, 2] - bboxes[:, 0])
    h = (bboxes[:, 3] - bboxes[:, 1])
    keep = (w >= 2) & (h >= 2)
    labels = labels[keep]
    K = K[keep]
    shapes = shapes[keep]
    bboxes = labels[:, 1:5]
    proj_2d, bboxes_2d = calc_proj2d_bbox3d(labels[:, 5:8], labels[:, 8:11], labels[:, -1], K)
    center = np.hstack([(bboxes[:, None, 0] + bboxes[:, None, 2])/2,
                        (bboxes[:, None, 1] + bboxes[:, None, 3])/2])

    # keep = (center[:, 0] >= 0) & (center[:, 0] < shapes[:, 0]) & (center[:, 1] >= 0) & (center[:, 1] < shapes[:, 1])
    # keep = (oc_min[:, 0] >= 0) & (oc_max[:, 0] < shapes[:, 0]) & (oc_min[:, 1] >= 0) & (oc_max[:, 1] < shapes[:, 1])
    iou = box_iou(bboxes, bboxes_2d)
    keep = iou > 0.4
    proj_2d = proj_2d[keep]
    center = center[keep]
    center = center.reshape((-1, 2, 1))
    offsets = proj_2d - center
    idx_max = np.argmax(offsets) // 18
    idx_min = np.argmin(offsets) // 18
    off_max = np.amax(offsets)
    off_min = np.amin(offsets)
    offset_max = offsets[idx_max]
    offset_min = offsets[idx_min]
    print(proj_2d[idx_max])
    print(proj_2d[idx_min])
    print(center[idx_max])
    print(center[idx_min])
    return off_max, off_min