import math
import os
import sys
import time
import cv2
import numpy as np
import torch
from image import save_image_batch_to_disk
from model import DexiNed


def app_path():
    """Returns the base application path."""
    if hasattr(sys, 'frozen'):
        # Handles PyInstaller
        return os.path.dirname(sys.executable)
    return os.path.dirname(__file__)


def usingDexiNed_multiple():
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

    #####################################################################
    # load data
    img_width = 512
    img_height = 512
    mean_bgr = [103.939, 116.779, 123.68]

    imagePath = os.path.join(root_path, 'pic')
    os.makedirs(imagePath, exist_ok=True)

    # the output path.
    output_dir = os.path.join(root_path, 'pic_result')
    os.makedirs(output_dir, exist_ok=True)

    # load the weight of the model to local device
    checkpoint_path = os.path.join(root_path, 'checkpoints/10_model.pth')
    os.makedirs(output_dir, exist_ok=True)

    model.load_state_dict(torch.load(checkpoint_path,
                                     map_location=device))
    model.eval()

    for imgId in range(1, 37):
        image = cv2.imread(os.path.join(imagePath, "00{:02d}.bmp".format(imgId)), cv2.IMREAD_COLOR)

        img = cv2.resize(image, (img_width, img_height))
        img = np.array(img, dtype=np.float32)
        img -= mean_bgr
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy()).float()
        inputImage = torch.unsqueeze(img, dim=0)

        targetImageShape = [torch.tensor([image.shape[0]]), torch.tensor([image.shape[1]])]
        # print(targetImageShape)
        file_names = "00{:02d}.bmp".format(imgId)

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


def combineImages():
    root_path = app_path()
    image_ori_path = os.path.join(root_path, 'pic')
    image_contour_path = os.path.join(root_path, 'pic_result')
    output_path = os.path.join(root_path, 'pic_result_combine')
    for imgId in range(1, 37):
        image_ori = cv2.imread(os.path.join(image_ori_path, "00{:02d}.bmp".format(imgId)), cv2.IMREAD_COLOR)
        image_contour = cv2.imread(os.path.join(image_contour_path, "00{:02d}.bmp".format(imgId)), cv2.IMREAD_COLOR)
        width = image_contour.shape[1]
        height = image_contour.shape[0]

        # get the most right LEFT point and the most left RIGHT point.
        lx, ly, rx, ry = modifyEdges(image_contour, 10)
        print(imgId, lx, ly, rx, ry)

        coverImage_left = np.zeros((ly, lx, 3))
        coverImage_right = np.zeros((ry, width - rx, 3))
        coverImage_left[:, :] = (255, 0, 0)
        coverImage_right[:, :] = (255, 0, 0)

        # get the black triangle based on the LEFT point.
        left_pt1 = (lx, height-ly)
        left_pt2 = (lx, height)
        left_pt3 = (lx+2*ly+1, height)  # to adjust with +1.
        left_triangle_cnt = np.array([left_pt1, left_pt2, left_pt3])

        # get the black triangle based on the RIGHT point.
        right_pt1 = (rx, height-ry)
        right_pt2 = (rx, height)
        right_pt3 = (rx-2*ry-1, height)       # to adjust with -1.
        right_triangle_cnt = np.array([right_pt1, right_pt2, right_pt3])

        image_contour[height - ly: height, : lx] = coverImage_left
        image_contour[height - ry: height, rx:] = coverImage_right

        # combine the contour image and the original image.
        # result_image = cv2.add(image_contour, image_ori)
        result_image = image_contour

        # add the left triangle and right triangle.
        cv2.drawContours(result_image, [left_triangle_cnt], 0, (255, 0, 0), -1)
        cv2.drawContours(result_image, [right_triangle_cnt], 0, (255, 0, 0), -1)

        cv2.imwrite(os.path.join(output_path, '00{:02d}.bmp'.format(imgId)), result_image)


# remove the bottom noisy of the original image.
def modifyEdges(edges, gap):
    height, width, _ = edges.shape
    temp = edges[height - gap:height, :]

    # get all point with color
    res = []
    for row in range(0, gap):
        # for col in range(0, width):
        for col in range(0, width):
            if temp[row][col][0] >= 30:
                res.append((col, row))
    res = sorted(res)
    # print(res)
    # keep the points existed in the same col, and separate them in groups.
    # such as [28, [0, 18]], where 28 is col, 0 and 18 is row.
    newRes = []
    pre = -1
    for i in range(0, len(res) - 1):
        if res[i][0] == pre:
            newRes.append(res[i])
        else:
            if res[i][0] == res[i + 1][0]:
                newRes.append(res[i])
            pre = res[i][0]
    all_values = [list[0] for list in newRes]
    unique_values = sorted(set(all_values))

    pre_result = []
    for value in unique_values:
        this_group = []
        for list in newRes:
            if list[0] == value:
                this_group.append(list[1])
        pre_result.append((value, this_group))

    # refine result (crop)
    temp = []
    for item in pre_result:
        x = item[0]
        y_list = item[1]
        if y_list[0] == 0:
            temp.append(x)
    crop_left = temp[0]
    crop_right = temp[len(temp)-1]

    result = []
    for item in pre_result:
        # print(item)
        x = item[0]
        y_list = item[1]
        if crop_left <= x <= crop_right and y_list[len(y_list)-1] == (gap-1):
            result.append((x, y_list))
    print("result")
    print(result)

    # get the most right LEFT point.
    point_y = 0
    # completeLeft = False
    left_first_x = 0
    left_first_y = 0
    find_first_left = False
    countLeftPointsWithGap2 = 0
    leftPoint_X = 0
    leftPoint_Y = 0
    for i in range(0, len(result)):
        item = result[i]
        X = item[0]     # col
        Y_list = item[1]    # row

        y, max_gap, stop = isContinuous(Y_list)
        # y, stop = isContinuous(Y_list, gap)
        point_y = y
        completeLeft = stop
        if completeLeft is True:           # find the intersection point.
            break

        if find_first_left is False and max_gap == 2:
            find_first_left = True
            left_first_x = X
            left_first_y = gap - point_y

        if find_first_left is True:
            countLeftPointsWithGap2 += 1

        if countLeftPointsWithGap2 == 10:
            leftPoint_Y = left_first_y
            leftPoint_X = left_first_x
            break
        else:
            leftPoint_Y = gap - point_y
            leftPoint_X = X

    # get the most left RIGHT point.
    result = sorted(result, reverse=True)
    print(result)

    right_first_x = 0
    right_first_y = 0
    find_first_right = False
    countRightPointsWithGap2 = 0
    rightPoint_X = 0
    rightPoint_Y = 0
    for j in range(0, len(result)):
        item = result[j]
        X = item[0]  # col
        Y_list = item[1]  # row
        y, max_gap, stop = isContinuous(Y_list)
        # y, stop = isContinuous(Y_list, gap)
        point_y = y
        completeRight = stop
        if completeRight is True:  # find the intersection point.
            break

        if find_first_right is False and max_gap == 2:
            find_first_right = True
            right_first_x = X
            right_first_y = gap - point_y

        if find_first_right is True:
            countRightPointsWithGap2 += 1

        if countRightPointsWithGap2 == 10:
            rightPoint_Y = right_first_y
            rightPoint_X = right_first_x
            break
        else:
            rightPoint_Y = gap - point_y
            rightPoint_X = X

    leftPoint_Y = min(leftPoint_Y, 4)
    rightPoint_Y = min(rightPoint_Y, 4)
    leftPoint_Y = max(leftPoint_Y, 3)
    rightPoint_Y = max(rightPoint_Y, 3)
    return leftPoint_X, leftPoint_Y, rightPoint_X, rightPoint_Y


def isContinuous(list):
    point_y = 0     # row
    max_gap = 1
    for i in reversed(range(1, len(list))):
        diff = list[i] - list[i - 1]
        if diff > 1:
            max_gap = max(max_gap, diff)
            point_y = list[i]
            if list[i] < 5:
                return point_y, max_gap, True
            else:
                break
    if max_gap == 1:
        return point_y, max_gap, True
    else:
        return point_y, max_gap, False


# def isContinuous(list, gap):
#     count = 5
#     point_y = 0
#     for i in reversed(range(1, len(list))):
#         if list[i] == (count + gap - 6):
#             if count > 0:
#                 count -= 1
#             if count == 0:
#                 point_y = 5
#                 break
#         else:
#             point_y = list[i+1]
#             return point_y, False
#     if count == 0:
#         return point_y, True
#     else:
#         return point_y, False
