import glob
import json
import numpy as np
import os
import tqdm
import cv2
import threading
import utils as common
from datasets.data.kitti.devkit_object import utils
from datasets.data.kitti.devkit_object import calib_utils
import math
root = '../'
out_dir = '../cache'


class myThread (threading.Thread):
    def __init__(self, threadID, callback, args):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.callback = callback
        self.args = args
        self._result = (None, None, None)

    def run(self):
        print("Start：" + str(self.threadID))
        self._result = self.callback(self.args)
        print("End：" + str(self.threadID))

    def get_result(self):
        return self._result


def multi_thread_load_dataset(files):
    sizes = []
    labels = []
    Ks = []
    for f in files:
        img_path = os.path.join(root, 'training', 'image_2', '{}.png'.format(f))
        label_path = os.path.join(root, 'training', 'label_2', '{}.txt'.format(f))
        calib_path = os.path.join(root, 'training', 'calib', '{}.txt'.format(f))

        # process shape
        img = cv2.imread(img_path)
        sizes.append(np.array([img.shape[1], img.shape[0]]))

        # process calib
        P2 = calib_utils.get_calib_P2(calib_path).reshape((3, -1))
        K = P2[:3, :3]
        T = P2[:3, 3]
        invK = cv2.invert(K, flags=cv2.DECOMP_LU)[1]
        TT = np.matmul(invK, T.reshape(-1, 1)).reshape((-1))
        with open(label_path) as f:
            objs = []
            for line in f.read().splitlines():
                splits = line.split()
                cls = utils.name_2_label(splits[0])
                if cls == -1:
                    continue
                cls = np.array([float(cls)])
                bbox = np.array([float(splits[4]), float(splits[5]), float(splits[6]), float(splits[7])])
                dim = np.array([float(splits[8]), float(splits[9]), float(splits[10])])  # H, W, L
                loc = np.array([float(splits[11]), float(splits[12]) - float(splits[8])/2, float(splits[13])]) + TT # x, y, z
                alpha = np.array([float(splits[3])])
                ry = np.array([float(splits[-1])])
                objs.append(np.concatenate([cls, bbox, dim, alpha, ry, loc], axis=0).reshape((1, -1)))
            labels.append(np.concatenate(objs, axis=0))
        Ks.append(K.reshape((-1)))

    return sizes, labels, Ks


def load_dataset(work_num=1, split='train'):
    th_pools = []
    with open(os.path.join(root, 'ImageSets', '{}.txt'.format(split))) as f:
        files = f.read().splitlines()
    files = sorted(files)
    # files = [files[i] for i in range(2000)]
    N = math.ceil(len(files) / work_num)
    for i in range(work_num):
        start = i * N
        end = min((i + 1) * N, len(files))
        th = myThread(i, multi_thread_load_dataset, [files[k] for k in range(start, end)])
        th_pools.append(th)

    for i in range(work_num):
        th_pools[i].start()
    for i in range(work_num):
        th_pools[i].join()

    sizes = []
    labels = []
    Ks = []
    for i in range(work_num):
        s, l, k = th_pools[i].get_result()
        sizes += s
        labels += l
        Ks += k
    path_shape = os.path.join(out_dir, 'shape_{}.npy'.format(split))
    path_label = os.path.join(out_dir, 'label_{}.npy'.format(split))
    path_k = os.path.join(out_dir, 'k_{}.npy'.format(split))
    np.save(path_shape, sizes)
    np.save(path_label, labels)
    np.save(path_k, Ks)
    return None


if __name__ == '__main__':
    load_dataset(work_num=10, split='test')
