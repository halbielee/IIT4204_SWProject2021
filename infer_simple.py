import os
import imageio
import argparse
import numpy as np
from PIL import Image

import torch
import torchvision
import torch.nn.functional as F
from util.imutils import HWC_to_CHW, Normalize
from metadata.dataset import load_img_id_list, load_img_label_list_from_npy
from network.resnet50_cls import Net


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", default="network.resnet50_cls", type=str)
    parser.add_argument("--weights", required=True, type=str)
    parser.add_argument("--infer_list", default="voc12/train.txt", type=str)
    parser.add_argument("--img_root", default='VOC2012', type=str)
    parser.add_argument("--cam_png", default=None, type=str)
    parser.add_argument("--thr", default=0.20, type=float)
    parser.add_argument("--dataset", default='voc12', type=str)
    args = parser.parse_args()

    if args.dataset == 'voc12':
        args.num_classes = 10

    # model information
    args.model_num_classes = args.num_classes

    # save path
    args.save_type = list()
    if args.cam_png is not None:
        os.makedirs(args.cam_png, exist_ok=True)
        args.save_type.append(args.cam_png)

    return args


def preprocess(image, scale_list, transform):
    img_size = image.size
    num_scales = len(scale_list)
    multi_scale_image_list = list()
    multi_scale_flipped_image_list = list()

    # insert multi-scale images
    for s in scale_list:
        target_size = (round(img_size[0] * s), round(img_size[1] * s))
        scaled_image = image.resize(target_size, resample=Image.CUBIC)
        multi_scale_image_list.append(scaled_image)
    # transform the multi-scaled image
    for i in range(num_scales):
        multi_scale_image_list[i] = transform(multi_scale_image_list[i])
    # augment the flipped image
    for i in range(num_scales):
        multi_scale_flipped_image_list.append(multi_scale_image_list[i])
        multi_scale_flipped_image_list.append(np.flip(multi_scale_image_list[i], -1).copy())
    return multi_scale_flipped_image_list


def predict_cam(model, image, label):

    original_image_size = np.asarray(image).shape[:2]
    multi_scale_flipped_image_list = preprocess(image, scales, transform)

    cam_list = list()
    model.eval()
    for i, image in enumerate(multi_scale_flipped_image_list):
        with torch.no_grad():
            image = torch.from_numpy(image).unsqueeze(0)
            image = image.cuda()
            cam = model.forward_cam(image)
            cam = F.interpolate(cam, original_image_size, mode='bilinear', align_corners=False)[0]
            cam = cam.cpu().numpy() * label.reshape(args.num_classes, 1, 1)
            if i % 2 == 1:
                cam = np.flip(cam, axis=-1)
            cam_list.append(cam)

    return cam_list


def infer_cam_mp(image_ids, label_list):
    print('process {} starts...'.format(os.getpid()))

    print('{} images per process'.format(len(image_ids)))
    model = Net(args.model_num_classes)
    model = model.cuda()
    model.load_state_dict(torch.load(args.weights))
    model.eval()

    with torch.no_grad():
        for i, (img_id, label) in enumerate(zip(image_ids, label_list)):
            # load image
            img_path = os.path.join(args.img_root, img_id + '.jpg')
            img = Image.open(img_path).convert('RGB')

            # infer cam_list
            cam_list = predict_cam(model, img, label)
            sum_cam = np.sum(cam_list, axis=0)
            norm_cam = sum_cam / (np.max(sum_cam, (1, 2), keepdims=True) + 1e-5)

            cam_dict = {}
            for j in range(args.num_classes):
                if label[j] > 1e-5:
                    cam_dict[j] = norm_cam[j]

            h, w = list(cam_dict.values())[0].shape
            tensor = np.zeros((args.num_classes + 1, h, w), np.float32)
            for key in cam_dict.keys():
                tensor[key + 1] = cam_dict[key]
            tensor[0, :, :] = args.thr
            pred = np.argmax(tensor, axis=0).astype(np.uint8)

            if args.cam_png is not None:
                imageio.imwrite(os.path.join(args.cam_png, img_id + '.png'), pred)

            if i % 10 == 0:
                print('{}/{} is complete'.format(i, len(image_ids)))


def main_mp():
    image_ids = load_img_id_list(args.infer_list)
    label_list = load_img_label_list_from_npy(image_ids, args.dataset)
    infer_cam_mp(image_ids, label_list)


if __name__ == '__main__':
    args = parse_args()

    scales = (0.5, 1.0, 1.5, 2.0)
    normalize = Normalize()
    transform = torchvision.transforms.Compose([np.asarray, normalize, HWC_to_CHW])

    main_mp()

