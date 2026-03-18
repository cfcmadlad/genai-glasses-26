import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import FILE_COL, LABEL_COL, RESIZED_DIR, TRAIN_CSV, TEST_CSV


TRAIN_TRANSFORMS = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.3),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

TEST_TRANSFORMS = A.Compose([
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])


class FacesDataset(Dataset):

    def __init__(self, resized_dir=RESIZED_DIR, train_csv=TRAIN_CSV, test_csv=TEST_CSV, augment=True):
        self.transforms = TRAIN_TRANSFORMS if augment else TEST_TRANSFORMS

        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)

        train_df["label"] = train_df[LABEL_COL].astype(int)
        test_df["label"] = -1

        train_df["filename"] = train_df[FILE_COL].apply(lambda x: f"face-{int(x)}.png")
        test_df["filename"] = test_df[FILE_COL].apply(lambda x: f"face-{int(x)}.png")

        combined = pd.concat([train_df[["filename", "label"]], test_df[["filename", "label"]]], ignore_index=True)
        self.data = combined.reset_index(drop=True)
        self.resized_dir = resized_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.resized_dir, row["filename"])

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        tensor = self.transforms(image=img)["image"]
        label = torch.tensor(row["label"], dtype=torch.long)

        return tensor, label