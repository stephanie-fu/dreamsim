from torch.utils.data import Dataset
from util.utils import get_preprocess_fn
from torchvision import transforms
import pandas as pd
import numpy as np
from PIL import Image
import os
from typing import Callable
import torch
import glob

IMAGE_EXTENSIONS = ["jpg", "png", "JPEG", "jpeg"]


class ThingsDataset(Dataset):
     def __init__(self, root_dir: str, txt_file: str, preprocess: str, load_size: int = 224,
                 interpolation: transforms.InterpolationMode = transforms.InterpolationMode.BICUBIC):
         with open(txt_file, "r") as f:
             self.txt = f.readlines()
         self.dataset_root = root_dir
         self.preprocess_fn = get_preprocess_fn(preprocess, load_size, interpolation)

     def __len__(self):
         return len(self.txt)

     def __getitem__(self, idx):
         im_1, im_2, im_3 = self.txt[idx].split()

         im_1 = Image.open(os.path.join(self.dataset_root, f"{im_1}.png"))
         im_2 = Image.open(os.path.join(self.dataset_root, f"{im_2}.png"))
         im_3 = Image.open(os.path.join(self.dataset_root, f"{im_3}.png"))

         im_1 = self.preprocess_fn(im_1)
         im_2 = self.preprocess_fn(im_2)
         im_3 = self.preprocess_fn(im_3)

         return im_1, im_2, im_3



class BAPPSDataset(Dataset):
     def __init__(self, root_dir: str, preprocess: str, load_size: int = 224,
                 interpolation: transforms.InterpolationMode = transforms.InterpolationMode.BICUBIC):
         data_types = ["cnn", "traditional", "color", "deblur", "superres", "frameinterp"]
         
         self.preprocess_fn = get_preprocess_fn(preprocess, load_size, interpolation)
         self.judge_paths = []
         self.p0_paths = []
         self.p1_paths = []
         self.ref_paths = []

         for dt in data_types:
             list_dir = os.path.join(os.path.join(root_dir, dt), "judge")
             for fname in os.scandir(list_dir):
                 self.judge_paths.append(os.path.join(list_dir, fname.name))
                 self.p0_paths.append(os.path.join(os.path.join(os.path.join(root_dir, dt), "p0"), fname.name.split(".")[0] + ".png"))
                 self.p1_paths.append(
                     os.path.join(os.path.join(os.path.join(root_dir, dt), "p1"), fname.name.split(".")[0] + ".png"))
                 self.ref_paths.append(
                     os.path.join(os.path.join(os.path.join(root_dir, dt), "ref"), fname.name.split(".")[0] + ".png"))

     def __len__(self):
         return len(self.judge_paths)

     def __getitem__(self, idx):
         judge = np.load(self.judge_paths[idx])
         im_left = self.preprocess_fn(Image.open(self.p0_paths[idx]))
         im_right = self.preprocess_fn(Image.open(self.p1_paths[idx]))
         im_ref = self.preprocess_fn(Image.open(self.ref_paths[idx]))
         return im_ref, im_left, im_right, judge

class DF2Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, split: str, preprocess: str, load_size: int = 224, 
                 interpolation: transforms.InterpolationMode = transforms.InterpolationMode.BICUBIC):
    
        self.preprocess_fn = get_preprocess_fn(preprocess, load_size, interpolation)
        # self.preprocess_fn=preprocess
        self.paths = get_paths(os.path.join(root_dir, split))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        im_path = self.paths[idx]
        img = Image.open(im_path)
        img = self.preprocess_fn(img)
        return img, im_path

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def get_paths(path):
    all_paths = []
    for ext in IMAGE_EXTENSIONS:
        all_paths += glob.glob(os.path.join(path, f"**.{ext}"))
    return all_paths
    
# class ImageDataset(torch.utils.data.Dataset):
#     def __init__(self, root, class_to_idx, transform=None, ret_path=False):
#         """
#         :param root: Dataset root. Should follow the structure class1/0.jpg...n.jpg, class2/0.jpg...n.jpg
#         :param class_to_idx: dictionary mapping the classnames to integers.
#         :param transform:
#         :param ret_path: boolean indicating whether to return the image path or not (useful for KNN for plotting nearest neighbors)
#         """

#         self.transform = transform
#         self.label_to_idx = class_to_idx

#         self.paths = []
#         self.labels = []
#         for cls in class_to_idx:
#             cls_paths = get_paths(os.path.join(root, cls))
#             self.paths += cls_paths
#             self.labels += [self.label_to_idx[cls] for _ in cls_paths]

#         self.ret_path = ret_path

#     def __len__(self):
#         return len(self.paths)

#     def __getitem__(self, idx):
#         im_path, label = self.paths[idx], self.labels[idx]
#         img = pil_loader(im_path)

#         if self.transform is not None:
#             img = self.transform(img)
#         if not self.ret_path:
#             return img, label
#         else:
#             return img, label, im_path
