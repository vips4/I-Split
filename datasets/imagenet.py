#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Federico Cunico, Luigi Capogrosso, Francesco Setti, \
              Damiano Carra, Franco Fummi, Marco Cristani"
__version__ = "1.0.0"
__maintainer__ = "Federico Cunico, Luigi Capogrosso"
__email__ = "name.surname@univr.it"


import os
import copy
import pickle

from tqdm import tqdm
from typing import Dict, Optional
from torch.utils.data import Dataset
from torchvision.datasets.folder import ImageFolder
from torchvision import transforms

# Hierarchy references:
#  https://observablehq.com/@mbostock/imagenet-hierarchy
#  https://gist.githubusercontent.com/mbostock/535395e279f5b83d732ea6e36932a9eb/raw/62863328f4841fce9452bc7e2f7b5e02b40f8f3d/mobilenet.json

class ImageNet(Dataset):
    def __init__(self, root: str, split: str, pre_load: bool = False) -> None:

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # trans = transforms.Compose([
        #     transforms.Resize((224, 224)),
        #     transforms.ToTensor(),
        #     normalize,
        # ])

        trans = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

        folder = os.path.join(root, split)
        # import h5py
        # fgfile = h5py.File(folder, 'r')

        file = f"{root}/database_{root.split(os.sep)[-1]}_{split}.pkl"
        if pre_load:
            if not os.path.isfile(file):
                print("Loading data...")
                res = []
                imf = ImageFolder(folder, trans)
                for data, gt_label in tqdm(imf):
                    res.append((data, gt_label))
                self._data = res
                with open(file, "wb") as fp:
                    pickle.dump(self._data, fp)
            else:
                with open(file, "rb") as fp:
                    self._data = pickle.load(fp)
        else:
            self._data = ImageFolder(folder, trans)

        print(f"Imagenet data loaded ({split}) {len(self._data)}")

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]


def get_imagenet_hierarchy(file_in: str, depth_level: Optional[int] = None, mode: str = "bfs"):
    import json
    file_out = file_in.split(".")[-1] + "_depth.json"

    if not os.path.isfile(file_out):
        with open(file_in, "r") as fp:
            data = json.load(fp)

        def assign_depth(_data, _idx):
            _data["depth"] = _idx
            if "children" in _data:
                children = _data["children"]
                for child in children:
                    assign_depth(child, _idx + 1)

        assign_depth(data, 0)

        with open(file_out, "w") as fp:
            json.dump(data, fp, indent=2)
    else:
        with open(file_out, "r") as fp:
            data = json.load(fp)
    if depth_level is None:
        depth_level = 0
    if mode == "dfs":
        return _imagenet_dfs(data, depth_level)
    elif mode == "bfs":
        return _imagenet_bfs(data, depth_level)
    else:
        raise NotImplementedError()


def flat_imagenet_classes(data):
    import copy

    els = []
    already_added = set()

    def _get_el(_data):
        candidate = copy.deepcopy(_data)
        if "children" in candidate:
            candidate.pop("children", None)
        return candidate

    def extensive_search(_data, _list):
        if _data["id"] not in already_added:
            candidate = _get_el(_data)
            already_added.add(_data["id"])
            els.append(candidate)

        if "children" in _data:
            children = _data["children"]
            for child in children:
                extensive_search(child, _list)

    extensive_search(data, els)
    return els


def _imagenet_bfs(data: Dict, depth_level: int):
    candidates = []

    def bfs(_data, _depth_level, _candidates):
        if _data["depth"] == _depth_level:
            _candidates.append(_data)
            return
        if "children" not in _data:
            return
        for child in _data["children"]:
            bfs(child, _depth_level, _candidates)

    bfs(data, depth_level, candidates)

    return candidates


def _imagenet_dfs(data: Dict, depth_level: int, include_lower: bool = True):
    import copy
    elements = []

    def _get_el(_data):
        candidate = copy.deepcopy(_data)
        if "children" in candidate:
            candidate.pop("children", None)
        return candidate

    def extensive_search(_data, _list):
        depth = _data["depth"]
        if depth >= depth_level:
            if depth == depth_level:  # the exact level
                candidate = _get_el(_data)
                _list.append(candidate)
            if include_lower and depth != depth_level:  # all the lower levels
                candidate = _get_el(_data)
                _list.append(_data)

        if "children" in _data:
            children = _data["children"]
            for child in children:
                extensive_search(child, _list)

    extensive_search(data, elements)

    return elements


def count_classes(data):

    # se è una foglia, ritorno
    if isinstance(data, dict):
        data = [data]

    uniques = set()  # classes are NOT unique!
    elements = []

    def _count(_e):
        if "children" not in _e:
            if _e["id"] in uniques:
                return 0
            else:
                uniques.add(_e["id"])
            tmp = copy.deepcopy(_e)
            if "children" in tmp:
                tmp.pop("children", None)
            elements.append(tmp)
            return 1
        val = 0
        for _child in _e["children"]:
            val += _count(_child)
        return val

    class_count = 0
    for e in data:
        class_count += _count(e)
    return class_count, elements


def filter_by(data, ids, names):
    print("Using deprecated method!")
    # DEPRECTATED
    data = data[0]  # ouch
    flat_data = flat_imagenet_classes(data)
    if ids is not None:
        els = [d for d in flat_data if d["id"] in ids]
    else:
        els = [d for d in flat_data if d["name"] in names]
    return els


def get_all_classes_with_common_ancestor(data: Dict, ancestor_name: Optional[str], ancestor_id: Optional[str] = None, ancestor_class: Optional[int] = None):
    if isinstance(data, list):
        if len(data) > 1:
            raise NotImplementedError()
        data = data[0]

    candidates = []

    def bfs(_data, _depth_level, ancestor_name, ancestor_id, ancestor_class):
        if len(candidates) > 0:
            return
        if ancestor_name is not None and ancestor_name in _data["name"]:
            candidates.append(_data)
            return
        if ancestor_id is not None and ancestor_id == _data["id"]:
            candidates.append(_data)
            return
        if ancestor_class is not None and ancestor_class == _data["index"]:
            candidates.append(_data)
            return
        if "children" not in _data:
            return
        for child in _data["children"]:
            bfs(child, _depth_level, ancestor_name, ancestor_id, ancestor_class)

    bfs(data, 0, ancestor_name, ancestor_id, ancestor_class)
    assert len(candidates) <= 1
    return candidates[0] if len(candidates) > 0 else None


def __test__():
    # from tqdm import tqdm
    # from torch.utils.data import DataLoader
    # dataset = ImageNet("data/imagenet", "train", pre_load=False)
    # dl = DataLoader(dataset, 280, False, num_workers=4)
    # for d in tqdm(dl):
    #     el_, class_ = d
    #     # print(class_)

    # dirs = os.listdir("./data/imagenet/val")
    elements = get_imagenet_hierarchy("./data/imagenet_classes_hierarchy.json", 13, mode="bfs")
    # indexes = [e["index"] for e in elements if e["id"] in dirs]
    # print(indexes)

    for i in range(15):
        elements = get_imagenet_hierarchy("./data/imagenet_classes_hierarchy.json", i)

        class_count, counted = count_classes(elements)
        assert len(counted) == class_count

        print(f"Depth: {i}  group number:  {len(elements)}   total number of classes: {class_count}")
        # for e in elements:
        #     print(f"  {e}")

    all_elements = get_imagenet_hierarchy("./data/imagenet_classes_hierarchy.json", mode="bfs")
    common = get_all_classes_with_common_ancestor(all_elements, ancestor_name="plant, works")
    common = get_all_classes_with_common_ancestor(all_elements, ancestor_name="natural object")


    print("end")

if __name__ == "__main__":
    __test__()
