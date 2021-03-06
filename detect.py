import argparse

# from utils.datasets import *
# from utils.utils import *
import cv2
import os
from pathlib import Path
import glob
# from alfred.dl.torch.common import device
import shutil
import torch
import time
import torchvision
import random
import numpy as np
from utils import metrics_utils
from datasets.kitti import LoadStreams
from utils.torch_utils import select_device
from models.yolo import Model
import pickle
from datasets.dataset_reader import DatasetReader
from preprocess.data_preprocess import  TestTransform
from postprocess import postprocess
import yaml
from utils import visual_utils


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(
        0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = max(img1_shape) / max(img0_shape)  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / \
            2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def detect(cfg, save_img=False):
    # out, source, weights, view_img, save_txt, imgsz = \
    #     opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    # webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize

    device = select_device(opt.device)

    half = device.type != 'cpu'  # half precision only supported on CUDA
    half = False
    # Load model
    model = torch.load(opt.weights, map_location=device)['model']
    model.to(device).eval()

    if half:
        model.half()  # to FP16
    else:
        model.to(torch.float32)

    # Set Dataloader
    dataset_path = cfg['dataset_path']
    dataset = DatasetReader(dataset_path, cfg, augment=TestTransform(cfg['img_size'][0], mean=cfg['brg_mean']),
                            is_training=False, split='test')
    # Get names and colors
    names = ['Car', 'Van', 'Truck', 'Pedestrian','Person_sitting', 'Cyclist', 'Tram', 'Misc']
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    encoder_decoder = model.model[-1].encoder_decoder
    # Run inference
    t0 = time.time()
    videowriter = None
    if cfg['write_video']:
        videowriter = cv2.VideoWriter('res.avi', cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'), 1, (1242, 750))
        max = 1000
        cnt = 0
    for img, targets, path, _ in dataset:
        src = cv2.imread(path)
        img = img.to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t_st = time.time()
        pred = model(img)[0]

        # Apply NMS
        pred = postprocess.apply_nms(pred, len(names), opt.conf_thres, opt.iou_thres, agnostic=opt.agnostic_nms)
        # decode pred
        bi = targets.get_field('img_id')
        K = targets.get_field('K')
        Ks = []
        for i in np.unique(bi):
            indices = i == bi
            Ks.append(K[indices][None, 0])
        Ks = np.concatenate(Ks, axis=0)
        pred = postprocess.decode_pred_logits(pred, (img.shape[3], img.shape[2]),
                                              [(src.shape[1], src.shape[0])], Ks, encoder_decoder)
        # postprocess.apply_batch_nms3d(pred)
        t_end = time.time()
        # print('pred after nms:', len(pred), pred[0].shape)
        mem = '%.3gG' % (torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
        print('cuda mem: %s ' % mem)
        src3d = np.copy(src)
        birdview = np.zeros((2*src.shape[0], src.shape[0], 3), dtype=np.uint8)
        if pred[0] is not None:
            src = visual_utils.cv_draw_bboxes_2d(src, pred[0], names)
            src3d = visual_utils.cv_draw_bboxes_3d(src3d, pred[0], names)
            birdview = visual_utils.cv_draw_bbox3d_birdview(birdview, pred[0], color=(255, 0, 0))
            birdview = visual_utils.cv_draw_bbox3d_birdview(birdview, targets, color=(0, 0, 255))
        concat_img = np.concatenate([src, src3d], axis=0)
        concat_img = np.concatenate([concat_img, birdview], axis=1)
        cv2.imwrite("TestOutputs/"+path[-10:-4]+".png",concat_img)
      #  cv2.imshow('test transform', concat_img)
        if cfg['write_video']:
            if cnt < max:
                concat_img = cv2.resize(concat_img, (1242, 750))
                # concat_img = concat_img[:, :, ::-1]
                videowriter.write(concat_img)
            cnt += 1

        print('the inference time of model is ', t_end - t_st)
        if cv2.waitKey(1000) == ord('q'):
            break
    if cfg['write_video']:
        videowriter.release()
    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/yolov5m.pt', help='model.pt path')
    parser.add_argument('--data', type=str, default='./datasets/configs/kitti.yaml', help='*.yaml path')
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size',  nargs='+', type=int, default=[640, 640], help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--is-mosaic', action='store_true', help='load image by applying mosaic')
    parser.add_argument('--is-rect', action='store_true', help='resize image apply rect mode not square mode')
    parser.add_argument('--write-video', action='store_true', help='write detect result to video')
    opt = parser.parse_args()
    # opt.img_size = check_img_size(opt.img_size)
    print(opt)
    cfg = opt.__dict__
    # dataset
    with open(cfg['data']) as f:
        data_cfg = yaml.load(f, Loader=yaml.FullLoader)  # data config
        cfg.update(data_cfg)
    with torch.no_grad():
        detect(cfg)
