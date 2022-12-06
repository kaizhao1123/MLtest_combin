import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class TestDataset(Dataset):
    def __init__(self,
                 data_root,
                 test_data,
                 mean_bgr,
                 img_height,
                 img_width,
                 test_list=None
                 ):
        self.data_root = data_root
        self.test_data = test_data
        self.test_list = test_list

        # self.arg = arg
        # self.mean_bgr = arg.mean_pixel_values[0:3] if len(arg.mean_pixel_values) == 4 \
        #     else arg.mean_pixel_values
        self.mean_bgr = mean_bgr
        self.img_height = img_height
        self.img_width = img_width
        self.data_index = self._build_index()

        print(f"mean_bgr: {self.mean_bgr}")
        # messagebox.showerror("dataset", "inial dataset")

    def _build_index(self):
        sample_indices = []
        # if self.test_data == "CLASSIC":

        # for single image testing
        images_path = os.listdir(self.data_root)
        labels_path = None
        sample_indices = [images_path, labels_path]

        # else:
        #     # image and label paths are located in a list file
        #
        #     if not self.test_list:
        #         raise ValueError(
        #             f"Test list not provided for dataset: {self.test_data}")
        #
        #     list_name = os.path.join(self.data_root, self.test_list)
        #     if self.test_data.upper()=='BIPED':
        #
        #         with open(list_name) as f:
        #             files = json.load(f)
        #         for pair in files:
        #             tmp_img = pair[0]
        #             tmp_gt = pair[1]
        #             sample_indices.append(
        #                 (os.path.join(self.data_root, tmp_img),
        #                  os.path.join(self.data_root, tmp_gt),))
        #     else:
        #         with open(list_name, 'r') as f:
        #             files = f.readlines()
        #         files = [line.strip() for line in files]
        #         pairs = [line.split() for line in files]
        #
        #         for pair in pairs:
        #             tmp_img = pair[0]
        #             tmp_gt = pair[1]
        #             sample_indices.append(
        #                 (os.path.join(self.data_root, tmp_img),
        #                  os.path.join(self.data_root, tmp_gt),))
        return sample_indices

    def __len__(self):
        return len(self.data_index[0]) if self.test_data.upper()=='CLASSIC' else len(self.data_index)

    def __getitem__(self, idx):
        # get data sample
        # image_path, label_path = self.data_index[idx]
        if self.data_index[1] is None:
            image_path = self.data_index[0][idx]
        else:
            image_path = self.data_index[idx][0]

        img_name = os.path.basename(image_path)
        file_name = os.path.splitext(img_name)[0] + ".png"

        img_dir = self.data_root
        gt_dir = None

        # load data
        image = cv2.imread(os.path.join(img_dir, image_path), cv2.IMREAD_COLOR)
        print("image_path")
        print(os.path.join(img_dir))
       # messagebox.showerror("dataset", os.path.join(img_dir, image_path))

        label = None

        im_shape = [image.shape[0], image.shape[1]]
        image, label = self.transform(img=image, gt=label)

        return dict(images=image, labels=label, file_names=file_name, image_shape=im_shape)

    def transform(self, img, gt):
        # gt[gt< 51] = 0 # test without gt discrimination
        if self.test_data == "CLASSIC":
            img_height = self.img_height
            img_width = self.img_width
            print(
                f"actual size: {img.shape}, target size: {( img_height,img_width,)}")
            img = cv2.resize(img, (img_width,img_height))
            gt = None

        img = np.array(img, dtype=np.float32)
        img -= self.mean_bgr
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy()).float()

        if self.test_data == "CLASSIC":
            gt = np.zeros((img.shape[:2]))
            gt = torch.from_numpy(np.array([gt])).float()
        # else:
        #     gt = np.array(gt, dtype=np.float32)
        #     if len(gt.shape) == 3:
        #         gt = gt[:, :, 0]
        #     gt /= 255.
        #     gt = torch.from_numpy(np.array([gt])).float()

        return img, gt

