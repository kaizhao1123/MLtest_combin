import math
import os
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from datasets import TestDataset
from image import save_image_batch_to_disk
from model import DexiNed
from tkinter import messagebox


# def test(checkpoint_path, dataloader, model, device, output_dir):
#     # if not os.path.isfile(checkpoint_path):
#     #     raise FileNotFoundError(
#     #         f"Checkpoint filte note found: {checkpoint_path}")
#     # print(f"Restoring weights from: {checkpoint_path}")
#
#     #messagebox.showerror("test func", "start")
#
#     with torch.no_grad():
#
#         #messagebox.showerror("test func", "start: 1")
#
#
#         for batch_id, sample_batched in enumerate(dataloader):
#
#             images = sample_batched['images'].to(device)
#             file_names = sample_batched['file_names']
#             image_shape = sample_batched['image_shape']
#             print("image info")
#             print(images)
#             print(type(images))
#             print("image shape")
#             print(image_shape)
#             print("image name")
#             print(file_names)
#             messagebox.showerror("test image :", file_names)
#
#             preds = model(images)   # get the tensor of the result image(contour)
#
#             # messagebox.showerror("preds", preds)
#
#             save_image_batch_to_disk(preds,
#                                      output_dir,
#                                      file_names,
#                                      image_shape
#                                      )
#             torch.cuda.empty_cache()


def usingDexiNed():

    def app_path():
        """Returns the base application path."""
        if hasattr(sys, 'frozen'):
            # Handles PyInstaller
            return os.path.dirname(sys.executable)
        return os.path.dirname(__file__)

    root_path = app_path()
    print(root_path)

    startTime = time.time()
    # Get computing device
    device = torch.device('cpu' if torch.cuda.device_count() == 0
                          else 'cuda')
    print("device: ")
    print(device)

    # Instantiate model and move it to the computing device
    model = DexiNed().to(device)

    # get the input image, transform it and save into torch dataloader.
    # dataset_val = TestDataset('data',
    #                           test_data='CLASSIC',
    #                           img_width=512,
    #                           img_height=512,
    #                           mean_bgr=[103.939, 116.779, 123.68],
    #                           test_list=None
    #                           )
    # dataloader_val = DataLoader(dataset_val,
    #                             batch_size=1,
    #                             shuffle=False,
    #                             num_workers=1)

    #####################################################################
    # load data
    img_width = 512
    img_height = 512
    mean_bgr = [103.939, 116.779, 123.68]

    imagePath = os.path.join(root_path, 'data')
    os.makedirs(imagePath, exist_ok=True)

    imgId = 13
    image = cv2.imread(os.path.join(imagePath, "00{:02d}.bmp".format(imgId)), cv2.IMREAD_COLOR)

    # messagebox.showerror("image", os.path.join(imagePath, "00{:02d}.bmp".format(1)))

    img = cv2.resize(image, (img_width, img_height))
    img = np.array(img, dtype=np.float32)
    img -= mean_bgr
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img.copy()).float()
    inputImage = torch.unsqueeze(img, dim=0)

    targetImageShape = [torch.tensor([image.shape[0]]), torch.tensor([image.shape[1]])]
    print(targetImageShape)
    file_names = "00{:02d}.png".format(imgId)

    #######################################################################3

    # the output path.
    output_dir = os.path.join(root_path, 'result')
    os.makedirs(output_dir, exist_ok=True)
    # messagebox.showerror("output_dir", output_dir)

    # load the weight of the model to local device
    checkpoint_path = os.path.join(root_path, 'checkpoints/10_model.pth')
    os.makedirs(output_dir, exist_ok=True)

    model.load_state_dict(torch.load(checkpoint_path,
                                     map_location=device))

    # messagebox.showerror("eval", "model")
    # Put model in evaluation mode
    model.eval()

    # targetImage = dataloader_val.dataset.__getitem__(0)
    # # print(torch.unsqueeze(targetImage['images'], dim=0))
    # inputImage = torch.unsqueeze(targetImage['images'], dim=0)
    # print(type(inputImage))

    # print(type(targetImage['image_shape']))
    # targetImageShape = [torch.tensor([targetImage['image_shape'][0]]), torch.tensor([targetImage['image_shape'][1]])]
    # print(targetImageShape)
    # file_names = [targetImage['file_names']]
    # print(file_names)
    #
    # # test the input image.

    with torch.no_grad():
        preds = model(inputImage)  # get the tensor of the result image(contour)
        save_image_batch_to_disk(preds,
                                 output_dir,
                                 file_names,
                                 targetImageShape
                                 )
        torch.cuda.empty_cache()

    # .__getitem__(0)
    # test(checkpoint_path, dataloader_val, model, device, output_dir)

    print("total time: ")
    print(time.time() - startTime)
    print('------------------- Test End -----------------------------')
    #messagebox.showerror("test info", "complete test! ")

    convertToMask(output_dir, file_names)


def convertToMask(output_dir, file_name):
    # convert to mask
    img = cv2.imread(os.path.join(output_dir, file_name))
    print(img.shape)
    # img = cv2.bitwise_not(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # change to gray image

    thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)[1]
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # Morphological denoising
    dst = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, element)
    contours = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    big_contour = max(contours, key=cv2.contourArea)
    maxRect = cv2.boundingRect(big_contour)
    X, Y, width, height = maxRect

    print(maxRect)

    # convert to binary image
    h, w, _ = img.shape
    bi = np.zeros([h, w, 3], dtype=np.uint8)
    cv2.drawContours(bi, [big_contour], 0, (255, 255, 255), thickness=-1)
    cv2.imwrite(output_dir + "/" + file_name + '_Thre_drawCon_binary.jpg', bi)

    # # 2. get the X, Y, width
    # gray_new = gray[0:gray.shape[0] - 10, 0:720]
    # thresh = cv2.threshold(gray_new, 0, 255, cv2.THRESH_BINARY)[1]
    # element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # Morphological denoising
    # dst = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, element)
    # contours = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    # big_contour = max(contours, key=cv2.contourArea)
    # maxRect = cv2.boundingRect(big_contour)
    # X, Y, width, _ = maxRect
    # print(maxRect)



    # crop the target and put it into the center of black template, to get the mask image.

    print(Y + height)
    print("**")
    firstCrop_binary = bi[Y: Y + height, X: X + width]
    start_Y = math.ceil((200 - height) / 2)
    start_X = math.ceil((200 - width) / 2)
    mask_image = np.zeros([200, 200, 3], dtype=np.uint8)

    print(start_Y + height)
    print("///")

    mask_image[start_Y: start_Y + height, start_X: start_X + width] = firstCrop_binary


    mask = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)  # array
    mask = Image.fromarray(mask)  # convert to image

    mask.save("C:/Users/Kai Zhao/PycharmProjects/MLtest2/result/" + 'Mask_0{:02d}0.png'.format(9 - 1))