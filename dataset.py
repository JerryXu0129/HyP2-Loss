import numpy as np
import collections

import torchvision
from torchvision import *
from xml.etree.ElementTree import Element as ET_Element
import os
from typing import Any, Callable, Dict, Optional
import torchvision.datasets.utils
import json
from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg
try:
    from defusedxml.ElementTree import parse as ET_parse
except ImportError:
    from xml.etree.ElementTree import parse as ET_parse

import torch
from PIL import Image, ImageFile
from torch.utils.data.dataset import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

class Flickr25k(Dataset):
    """
    Flicker 25k dataset.

    Args
        root(str): Path of dataset.
        mode(str, 'train', 'query', 'retrieval'): Mode of dataset.
        transform(callable, optional): Transform images.
    """
    def __init__(self, root, mode, transform=None):
        self.root = root
        self.transform = transform
        # self.diff = None
        if mode == 'train':
            self.data = [Image.open(os.path.join(root, 'mirflickr', i)).convert('RGB') for i in Flickr25k.TRAIN_DATA]
            self.targets = Flickr25k.TRAIN_TARGETS
            # self.targets.dot(self.targets.T) == 0
        elif mode == 'query':
            self.data = [Image.open(os.path.join(root, 'mirflickr', i)).convert('RGB') for i in Flickr25k.QUERY_DATA]
            self.targets = Flickr25k.QUERY_TARGETS
        elif mode == 'retrieval':
            self.data = [Image.open(os.path.join(root, 'mirflickr', i)).convert('RGB') for i in Flickr25k.RETRIEVAL_DATA]
            self.targets = Flickr25k.RETRIEVAL_TARGETS
        else:
            raise ValueError(r'Invalid arguments: mode, can\'t load dataset!')

    def __getitem__(self, index):
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, self.targets[index]
    def __len__(self):
        return len(self.data)

    def get_targets(self):
        return torch.FloatTensor(self.targets)

    @staticmethod
    def init(root, num_query, num_train):
        # Load dataset
        img_txt_path = os.path.join(root, 'img.txt')
        targets_txt_path = os.path.join(root, 'targets.txt')

        # Read files
        with open(img_txt_path, 'r') as f:
            data = np.array([i.strip() for i in f])
        targets = np.loadtxt(targets_txt_path, dtype=np.int64)

        # Split dataset
        perm_file = 'flickr.txt'
        if os.path.exists(perm_file):
            perm_index = np.array(json.loads(open(perm_file, 'r').read()))
            print('------------- flickr loaded -------------')
        else:
            perm_index = np.random.permutation(data.shape[0]).tolist()
            flickr_txt = open(perm_file, 'w')
            flickr_txt.write(json.dumps(perm_index))
            flickr_txt.close()
            print('------------- flickr initialized -------------')
        
        query_index = perm_index[:num_query]
        train_index = perm_index[num_query: num_query + num_train]
        retrieval_index = perm_index[num_query + num_train:]

        Flickr25k.QUERY_DATA = data[query_index]
        Flickr25k.QUERY_TARGETS = targets[query_index, :]

        Flickr25k.TRAIN_DATA = data[train_index]
        Flickr25k.TRAIN_TARGETS = targets[train_index, :]

        Flickr25k.RETRIEVAL_DATA = data[retrieval_index]
        Flickr25k.RETRIEVAL_TARGETS = targets[retrieval_index, :]

class ImageList(object):
    def __init__(self, image_list, labels=None, transform=None):
        self.imgs = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(open('./data/nus_wide/' + path, 'rb')).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.imgs)

DATASET_YEAR_DICT = {
    "2012": {
        "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
        "filename": "VOCtrainval_11-May-2012.tar",
        "md5": "6cd6e144f989b92b3379bac3b3de84fd",
        "base_dir": os.path.join("VOCdevkit", "VOC2012"),
    },
    "2007": {
        "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar",
        "filename": "VOCtrainval_06-Nov-2007.tar",
        "md5": "c52e279531787c972589f7e41ab4ae64",
        "base_dir": os.path.join("VOCdevkit", "VOC2007"),
    },
    "2007-test": {
        "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar",
        "filename": "VOCtest_06-Nov-2007.tar",
        "md5": "b6e924de25625d8de591ea690078ad9f",
        "base_dir": os.path.join("VOCdevkit", "VOC2007"),
    },
}

VOC_labels = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4, 'bus': 5, 'car': 6, 'cat': 7, 'chair': 8, 'cow': 9, 'diningtable': 10, 'dog': 11, 'horse': 12, 'motorbike': 13, 'person': 14, 'pottedplant': 15, 'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19}

class VOCBase(torchvision.datasets.VisionDataset):
    _SPLITS_DIR = "Main"
    _TARGET_DIR = "Annotations"
    _TARGET_FILE_EXT = ".xml"

    def __init__(
        self,
        root: str,
        year: str = "2012",
        image_set: str = "train",
        download: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        super().__init__(root, transforms, transform, target_transform)
        self.year = year

        valid_image_sets = ["train", "trainval", "val", "TRAIN", "VAL", "DATABASE"]
        if year == "2007":
            valid_image_sets.append("test")
        self.image_set = verify_str_arg(image_set, "image_set", valid_image_sets)
        key = "2007-test" if year == "2007" and image_set == "test" else year
        dataset_year_dict = DATASET_YEAR_DICT[key]
        self.url = dataset_year_dict["url"]
        self.filename = dataset_year_dict["filename"]
        self.md5 = dataset_year_dict["md5"]
        base_dir = dataset_year_dict["base_dir"]
        voc_root = os.path.join(self.root, base_dir)

        if download:
            download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.md5)

        if not os.path.isdir(voc_root):
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        splits_dir = os.path.join(voc_root, "ImageSets", self._SPLITS_DIR)
        split_f = os.path.join(splits_dir, image_set.rstrip("\n") + ".txt")
        with open(os.path.join(split_f)) as f:
            file_names = [x.strip() for x in f.readlines()]

        image_dir = os.path.join(voc_root, "JPEGImages")
        self.images = [Image.open(os.path.join(image_dir, x + ".jpg")).convert("RGB") for x in file_names]

        target_dir = os.path.join(voc_root, self._TARGET_DIR)
        self.targets = []
        for x in file_names:
            target = self.parse_voc_xml(ET_parse(os.path.join(target_dir, x + self._TARGET_FILE_EXT)).getroot())
            labels = tuple([i['name'] for i in target['annotation']['object']])
            target = np.zeros(20)
            for i in labels:
                target[VOC_labels[i]] = 1
            self.targets.append(target)
        self.targets = np.array(self.targets)
        assert len(self.images) == len(self.targets)
        self.stat = self.targets.sum(axis = 0)

    def __getitem__(self, index: int):
        img = self.images[index]
        target = self.targets[index]
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.images)

    @staticmethod
    def parse_voc_xml(node: ET_Element):
        voc_dict: Dict[str, Any] = {}
        children = list(node)
        if children:
            def_dic: Dict[str, Any] = collections.defaultdict(list)
            for dc in map(VOCBase.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == "annotation":
                def_dic["object"] = [def_dic["object"]]
            voc_dict = {node.tag: {ind: v[0] if len(v) == 1 else v for ind, v in def_dic.items()}}
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict
