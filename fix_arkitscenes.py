# -*- coding: utf-8 -*-
"""mde_arkit_metric_anything.ipynb
"""

import numpy as np
import cv2
import gc
import os
import shutil
import subprocess

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


from pathlib import Path
# from PIL import Image
from tqdm.notebook import tqdm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2


print('ok')

show_imgs = False
name = 'Metric_Anything'

"""### helping fucntions"""

def print_dir_struct(path: Path, max_depth=3, max_cnt=5, prefix="", cur_depth=0):
    items = os.listdir(path)
    if (len(items) > max_cnt):
        items = items[:max_cnt] + [". . ."]
    for i, item in enumerate(items):
        custom_prefix = "└── " if i == len(items) - 1 else "├── "
        print(prefix, custom_prefix, item)
        if cur_depth < max_depth and os.path.isdir(os.path.join(path, item)) and item != ". . .":
            custom_prefix = '    ' if i == len(items) - 1 else ' │   '
            print_dir_struct(os.path.join(path, item), max_depth, max_cnt, prefix + custom_prefix, cur_depth + 1)

def show_res(rgb, lr_depth, hr_depth, depth, name='tmp'):
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    hr_depth[hr_depth == 0] = np.nan
    data = [rgb, lr_depth, hr_depth, depth]
    titles = ["RGB", "LR Depth", "HR Depth", "Predicted Depth"]
    for i in range(4):
        im = ax[i // 2, i % 2].imshow(data[i])
        ax[i // 2, i % 2].set_title(titles[i])
        fig.colorbar(im, ax=ax[i // 2, i % 2])
    plt.tight_layout()
    plt.savefig(name+".png")
    plt.show()

"""### ARKitScenes

!git clone https://github.com/apple/ARKitScenes.git

MAX DEPTH LR = 5 meters !
"""

def check_dir_for_cleanup(path='.', max_gb=10.0, target_gb=0.75):
    MAX_BYTES = max_gb * (1024**3)
    TARGET_BYTES = target_gb * (1024**3)
    def get_dir_size_fast(path):
        total = 0
        with os.scandir(path) as it:
            for entry in it:
                if entry.is_file():
                    total += entry.stat().st_size
                elif entry.is_dir():
                    total += get_dir_size_fast(entry.path)
        return total

    with os.scandir(path) as it:
        files = [entry for entry in it]
    files.sort(key=lambda x: x.stat().st_mtime)
    files_size = np.array([file.stat().st_size if file.is_file() else get_dir_size_fast(file.path) for file in files])
    if np.sum(files_size) > MAX_BYTES:

        deleted = 0
        for file, size in zip(files, files_size):
            if file.is_file():
                os.remove(file.path)
            elif file.is_dir():
                shutil.rmtree(file.path)
            deleted += size
            if deleted >= TARGET_BYTES:
                return 0
    return 0

def get_sorted_dirs(path='.'):
    with os.scandir(path) as it:
        dirs = [entry for entry in it if entry.is_dir()]
    dirs.sort(key=lambda x: x.stat().st_mtime)
    return dirs

class ARKitDataset:
    def __init__(self,
    download_dir= os.path.join(os.getcwd(), 'datasets', 'arkit-scenes'),
    git_repository_path=os.path.join(os.getcwd(), 'ARKitScenes'),
    dataset='upsampling',
    split='Training',
    silent=False):
        self.download_dir = download_dir
        self.git_repository_path = git_repository_path
        self.dataset = dataset
        self.split = split
        self.silent = silent
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)
        train_val_splits_df = pd.read_csv(os.path.join(git_repository_path, "depth_upsampling", "upsampling_train_val_splits.csv"))

        self.video_ids = train_val_splits_df[train_val_splits_df['fold'] == split]['video_id'].to_list()

        self.data_dir = os.path.join(self.download_dir, dataset, self.split)
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def run_loading_data_script(
            self,
    csv_path=None,
    video_ids=None,
    ):
        cmd = [
            'python',
            os.path.join(self.git_repository_path, 'download_data.py'),
            self.dataset,
            '--split', self.split,
            '--download_dir', self.download_dir,
            '--raw_dataset_assets', 'lowres_wide_intrinsics',
        ]
        if (csv_path is not None):
            cmd += ['--video_id_csv'] + [csv_path]
        elif (video_ids is not None):
            cmd += ['--video_id'] + list(map(str, video_ids))
            video_paths = list()
            for video_id in video_ids:
                video_path = os.path.join(self.download_dir, self.dataset, self.split, video_id)
                video_paths.append(video_path)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if (not self.silent): print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
        return self.download_dir

    def image_seq(self, n=None, skip_to=None):
        scene_count = len(self.video_ids)
        img_returned = 0
        img_limit = False
        start = 0 if skip_to is None else self.video_ids.index(skip_to)
        for i in range(start, scene_count):
            # check if space is avaible
            # check_dir_for_cleanup(self.data_dir)

            self.run_loading_data_script(video_ids=[str(self.video_ids[i])])

            video_dir = os.path.join(self.data_dir, str(self.video_ids[i]))
            hr_depth_dir = os.path.join(video_dir, "highres_depth")
            lr_depth_dir = os.path.join(video_dir, "lowres_depth")
            rgb_dir = os.path.join(video_dir, "wide")
            for filename in os.listdir(hr_depth_dir):
                hr_depth_np = cv2.imread(os.path.join(hr_depth_dir, filename), cv2.IMREAD_UNCHANGED)
                lr_depth_np = cv2.imread(os.path.join(lr_depth_dir, filename), cv2.IMREAD_UNCHANGED)

                # Convert to Torch Tensors and Scale (keeping on GPU in float16)
                hr_depth = torch.from_numpy(hr_depth_np.astype('float32')).to(DEVICE) / 1000.0
                lr_depth = torch.from_numpy(lr_depth_np.astype('float32')).to(DEVICE) / 1000.0

                # 2. Load and Prepare RGB
                rgb_np = cv2.imread(os.path.join(rgb_dir, filename)) # BGR by default
                rgb = cv2.cvtColor(rgb_np, cv2.COLOR_BGR2RGB)     # Convert to RGB

                # Convert to [C, H, W] format and Scale to 0-1
                # rgb = torch.from_numpy(rgb_np).to(DEVICE).permute(2, 0, 1).float() / 255.0

                if self.video_ids[i] == 48458647: # damaged scene
                    hr_depth = torch.rot90(hr_depth, k=-1, dims=(1, 2))
                    lr_depth = torch.rot90(lr_depth, k=-1, dims=(1, 2))


                yield rgb, lr_depth, hr_depth
                img_returned += 1
                if (n is not None) and (img_returned >= n):
                    img_limit = True
                    break
            if img_limit:
                break
    def all_download(self, n=None, skip_to=None):
            scene_count = len(self.video_ids)
            img_returned = 0
            img_limit = False
            start = 0 if skip_to is None else self.video_ids.index(skip_to)
            for i in range(start, scene_count):
                # check if space is avaible
                # check_dir_for_cleanup(self.data_dir)
                if self.video_ids[i] == 48458647:
                    continue
                video_dir = os.path.join(self.data_dir, str(self.video_ids[i]))
                if not os.path.exists(video_dir):
                    self.run_loading_data_script(video_ids=[str(self.video_ids[i])])

                if self.dataset == 'unsampling':
                    rgb_dir = os.path.join(video_dir, "wide")
                    for filename in os.listdir(rgb_dir):
                        # if self.video_ids[i] == 48458647: # damaged scene
                        #     rgb_path = os.path.join(rgb_dir, filename)
                        #     rgb_img = cv2.imread(rgb_path)
                        #     rgb_img = np.rot90(rgb_img, k=-1, axes=(0, 1))
                        #     cv2.imwrite(rgb_path, rgb_img)
                        img_returned += 1
                        if (n is not None) and (img_returned >= n):
                            img_limit = True
                            break
                    if img_limit:
                        break


split='Validation'

print('Download begin...')

dataset_downloader = ARKitDataset(split=split)

id = str(48458647)

dataset_downloader.run_loading_data_script(video_ids=[id])

video_dir = os.path.join(dataset_downloader.data_dir, id)

rgb_dir = os.path.join(video_dir, "wide")
for filename in os.listdir(rgb_dir):
    rgb_path = os.path.join(rgb_dir, filename)
    rgb_img = cv2.imread(rgb_path)
    if rgb_img is not None:
        print(rgb_img.shape, ' is shape')
        # Rotate 90 degrees clockwise
        rgb_img = np.rot90(rgb_img, k=1, axes=(0, 1))
        # Save back to the same path
        cv2.imwrite(rgb_path, rgb_img)
    else:
        print(f"Failed to read: {rgb_path}")

