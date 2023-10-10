import math
import os
import cv2
import numpy as np
import torch
from PIL import Image
# from tkinter import messagebox

def image_normalization(img, img_min=0, img_max=255, epsilon=1e-12):
    """This is a typical image normalization function
    where the minimum and maximum of the image is needed
    source: https://en.wikipedia.org/wiki/Normalization_(image_processing)
    :param img: an image could be gray scale or color
    :param img_min:  for default is 0
    :param img_max: for default is 255
    :return: a normalized image, if max is 255 the dtype is uint8
    """

    img = np.float32(img)
    # whenever an inconsistent image
    img = (img - np.min(img)) * (img_max - img_min) / \
        ((np.max(img) - np.min(img)) + epsilon) + img_min
    return img


def count_parameters(model=None):
    if model is not None:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        print("Error counting model parameters line 32 img_processing.py")
        raise NotImplementedError


def save_image_batch_to_disk(tensor, output_dir, file_name, img_shape=None):

    # os.makedirs(output_dir, exist_ok=True)

    edge_maps = []
    for i in tensor:
        tmp = torch.sigmoid(i).cpu().detach().numpy()
        edge_maps.append(tmp)
    tensor = np.array(edge_maps)

    image_shape = [x.cpu().detach().numpy() for x in img_shape]
    # (H, W) -> (W, H)
    image_shape = [[y, x] for x, y in zip(image_shape[0], image_shape[1])]

    i_shape = image_shape[0]
    tmp = tensor[:, 0, ...]
    tmp = np.squeeze(tmp)

    # Iterate our all 7 NN outputs for a particular image
    # for i in range(tmp.shape[0]):
    #     tmp_img = tmp[i]
    #     tmp_img = np.uint8(image_normalization(tmp_img))
    #     # Resize prediction to match input image size
    #     if not tmp_img.shape[1] == i_shape[0] or not tmp_img.shape[0] == i_shape[1]:
    #         tmp_img = cv2.resize(tmp_img, (i_shape[0], i_shape[1]))
    #
    #     if i == 6:
    #         fuse = tmp_img
    #         fuse = fuse.astype(np.uint8)

    # there are 7 NN outputs, we need the last one.
    tmp_img = tmp[len(tmp)-1]
    tmp_img = np.uint8(image_normalization(tmp_img))

    # Resize prediction to match input image size
    if not tmp_img.shape[1] == i_shape[0] or not tmp_img.shape[0] == i_shape[1]:
        tmp_img = cv2.resize(tmp_img, (i_shape[0], i_shape[1]))

    fuse = tmp_img
    fuse = fuse.astype(np.uint8)

    output_file_name_f = os.path.join(output_dir, file_name)
    cv2.imwrite(output_file_name_f, fuse)

    # #########################
