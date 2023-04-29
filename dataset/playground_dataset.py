import os
from typing import List, Union
from torch.utils.data import Dataset
from torchvision import transforms
import utils
from PIL import Image
import numpy as np
import torch


class PlaygroundTrainDataset(Dataset):
    def __init__(self, root, mode='train', num_classes=1, transform=None):
        """
        Args:
            root:
            mode (str): 可选'train'、'validate'
            num_classes:
            transform:
        """
        super(PlaygroundTrainDataset, self).__init__()
        if not transform:
            transform = transforms.ToTensor()
        self.transform = transform
        self.num_classes = num_classes

        image_root = os.path.join(root, mode, 'image')
        mask_root = os.path.join(root, mode, 'mask')

        # self.img_list = []
        # self.mask_list = []
        self._img = []
        self._mask = []
        for image_dir, mask_dir in zip(os.listdir(image_root), os.listdir(mask_root)):
            img_patient = os.path.join(image_root, image_dir)
            mask_patient = os.path.join(mask_root, mask_dir)
            for img_name, mask_name in zip(os.listdir(img_patient), os.listdir(mask_patient)):
                self._img.append(os.path.join(img_patient, img_name))
                self._mask.append(os.path.join(mask_patient, mask_name))
        print(f'Load {len(self)} samples.')

    def __getitem__(self, idx):
        img = Image.open(self._img[idx])
        img = self.transform(img) if self.transform is not None else img

        resize_size = tuple(img.shape[-2:])
        mask = Image.open(self._mask[idx])
        mask = mask.resize(resize_size)
        mask = np.array(mask, dtype=np.float32)
        mask = torch.from_numpy(mask).long()
        return img, mask

    # def __getitem__(self, idx):
    #     img = Image.open(self._img[idx])
    #     img = self.transform(img) if self.transform is not None else img
    #     mask = Image.open(self._mask[idx])
    #     mask = self.transform(mask) if self.transform is not None else mask
    #     return img, mask

    def __len__(self):
        return len(self._img)


if __name__ == '__main__':
    from torchvision import transforms

    # img_trans = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Resize((512, 512))
    # ])
    # mask_trans = img_trans
    # train_data = PlaygroundTrainDataset(root='E:/all_projects/data/MICCAI_pre_test_data', mode='train', transform=img_trans)
    train_data = PlaygroundTrainDataset(root='E:/all_projects/data/MICCAI_pre_test_data', mode='train')

    a, b = train_data[20]
    temp = 0
