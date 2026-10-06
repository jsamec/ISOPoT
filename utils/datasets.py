import os
import json

import numpy as np
import cv2

from torch.utils.data import Dataset

from models.superpoint import detection as det


class ISOPoTDataset(Dataset):
    def __init__(
        self, root_path, filter_keypoints=True, vflip_image=False,
        hflip_image=False, generate_kpts=False, n=1, superpoint_conf=0.04, save_kpts=False, return_first_kpts_only=True
    ):
        self.root_path = root_path
        self.filter_keypoints = filter_keypoints
        self.vflip_image = vflip_image
        self.hflip_image = hflip_image
        self.generate_kpts = generate_kpts
        self.n = n
        self.superpoint_conf = superpoint_conf
        self.save_kpts = save_kpts
        self.return_first_kpts_only = return_first_kpts_only

        self.indexes = self._gather_indexes()

    def _gather_indexes(self):
        files = os.listdir(self.root_path)
        indexes = []
        for file in files:
            if file.endswith(".png"):
                idx = file.split(".")[0]
                if f"{idx}.txt" not in files:
                    raise FileNotFoundError(f"Pose file for {idx} not found in {self.root_path}")
                indexes.append(int(idx))

        indexes = list(set(indexes))  # Remove duplicates
        indexes = sorted(indexes)  # Sort the indexes
        print(f"Found {len(indexes)} valid image-pose pairs in {self.root_path}.")
        return indexes

    def __len__(self):
        return len(self.indexes) - self.n + 1  # so we don't go out of bounds

    def __getitem__(self, idx):
        if idx + self.n > len(self.indexes):
            raise IndexError("Index out of range for dataset.")

        images, keypoints_list, poses = [], [], []

        for offset in range(self.n):
            true_idx = self.indexes[idx + offset]
            image = self._load_image(true_idx)
            kpts = self._load_keypoints(true_idx, image)
            pose = self._load_pose(true_idx)

            if self.filter_keypoints:
                kpts = self._filter_keypoints(image, kpts)

            images.append(image)
            keypoints_list.append(kpts)
            poses.append(pose)

        return images, poses, keypoints_list[0] if self.return_first_kpts_only else keypoints_list

    def _load_image(self, idx):
        image_path = os.path.join(self.root_path, f"{idx}.png")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file {image_path} not found.")

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if self.vflip_image:
            image = cv2.flip(image, 0)
        if self.hflip_image:
            image = cv2.flip(image, 1)

        return image

    def _load_pose(self, idx):
        pose_path = os.path.join(self.root_path, f"{idx}.txt")
        if not os.path.exists(pose_path):
            raise FileNotFoundError(f"Pose file {pose_path} not found.")
        with open(pose_path, 'r') as f:
            return json.load(f)

    def _load_keypoints(self, idx, image):
        if self.generate_kpts:
            # Replace with your actual detection function
            kpts = det.generate_query_kpts(image, h=image.shape[0], w=image.shape[1], superpoint_conf=self.superpoint_conf)[0]

            #save keypoints to file
            if self.save_kpts:
                kpt_path = os.path.join(self.root_path, f"{idx}_c:{self.superpoint_conf}.npy")
                np.save(kpt_path, kpts)

        else:
            kpt_path = os.path.join(self.root_path, f"{idx}_c:{self.superpoint_conf}.npy")
            if not os.path.exists(kpt_path):
                return None
            kpts = np.load(kpt_path)

            if self.vflip_image and kpts.shape[0] > 0:
                kpts[:, 1] = image.shape[0] - kpts[:, 1]
            if self.hflip_image and kpts.shape[0] > 0:
                kpts[:, 0] = image.shape[1] - kpts[:, 0]

        return kpts

    def _filter_keypoints(self, image, kpts):
        if kpts == None:
            return None
        mask = cv2.threshold(image, 0, 255, 0)[1]
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
        mask = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=5)

        filtered_kpts = [kpt for kpt in kpts if mask[int(kpt[1]), int(kpt[0])] > 0]
        return np.array(filtered_kpts, dtype=np.float32)
