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

print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
torch.cuda.empty_cache()

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

from torch.utils.data import Dataset, DataLoader
import cv2

class ARKitTrainDataset(Dataset):
    def __init__(self, data_dir=os.path.join(os.getcwd(), 'datasets', 'arkit-scenes'), split='Training', transform=None):
        self.data_dir = os.path.join(data_dir, 'upsampling', split)
        self.f_px_dir = os.path.join(data_dir, 'raw', split)
        self.transform = transform
        self.samples = []

        # dataset_downloader = ARKitDataset(split=split)
        # dataset_downloader.all_download()

        # Собираем пути ко всем кадрам заранее
        # Ожидаем структуру: data_dir/video_id/wide/*.png
        for video_id in os.listdir(self.data_dir):
            rgb_path = os.path.join(self.data_dir, video_id, "wide")
            if os.path.isdir(rgb_path):
                for fname in os.listdir(rgb_path):
                    self.samples.append({
                        "video_id": video_id,
                        "file": fname
                    })
        # print(len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # print('read')
        sample = self.samples[idx]
        video_dir = os.path.join(self.data_dir, sample["video_id"])
        video_f_px_dir = os.path.join(self.f_px_dir, sample["video_id"])

        # Загрузка
        rgb = cv2.imread(os.path.join(video_dir, "wide", sample["file"]))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        hr_depth = cv2.imread(os.path.join(video_dir, "highres_depth", sample["file"]), cv2.IMREAD_UNCHANGED)
        hr_depth = hr_depth.astype('float32') / 1000.0

        lr_depth = cv2.imread(os.path.join(video_dir, "lowres_depth", sample["file"]), cv2.IMREAD_UNCHANGED)
        lr_depth = lr_depth.astype('float32') / 1000.0

        pincam_file = os.path.join(video_f_px_dir, 'lowres_wide_intrinsics', sample["file"]).replace('.png', '.pincam')
        f_px = -1460.0 # Значение по умолчанию, если файл не найден
        # print(pincam_file, os.path.exists(pincam_file))
        if os.path.exists(pincam_file):
            with open(pincam_file, 'r') as f:
                data = f.read().split()
                # Для lowres (256x192) fx это data[2], fy это data[3]
                # Масштабируем до highres (1920x1440), умножая на 7.5
                f_px = float(data[2]) * 7.5

        # Обязательно приведите к формату [C, H, W] для PyTorch
        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        depth_tensor = torch.from_numpy(hr_depth).unsqueeze(0) # [1, H, W]
        low_depth_tensor = torch.from_numpy(lr_depth).unsqueeze(0) # [1, H?, W?]


        if self.transform:
            rgb_tensor = self.transform(rgb_tensor)

        return rgb_tensor, depth_tensor, low_depth_tensor, f_px

"""### Metric Anything"""

import sys
from torchvision.transforms import v2

lib_path = os.path.join(os.getcwd(), 'metric-anything', 'models', 'student_depthmap')
if lib_path not in sys.path:
    sys.path.append(lib_path)

from depth_model import MetricAnythingDepthMap

original_dir = os.getcwd()
os.chdir(lib_path)

checkpoint_file = os.path.join(original_dir, "student_depthmap.pt")

model = MetricAnythingDepthMap.from_pretrained(
   checkpoint_file
)
model = model.to(DEVICE).eval()

os.chdir(original_dir)

transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)),
])

def predict0(depth, lr_depth):
    return depth

def predict1_batch(depth, lr_depth):
    """
    Least Squares Alignment (s*d + t) for a Batch.
    Ax = B solved via Normal Equations for speed on GPU.
    """
    B, C, H, W = depth.shape
    # Resize depth to match lr_depth for the math comparison
    d_res = F.interpolate(depth, size=lr_depth.shape[-2:], mode='bilinear', align_corners=False)

    # Flatten spatial dims: [B, N]
    x = d_res.view(B, -1)
    y = lr_depth.view(B, -1)

    # We need to solve s*x + t = y for each item in batch
    # Compute components for linear regression:
    x_mean = x.mean(dim=1, keepdim=True)
    y_mean = y.mean(dim=1, keepdim=True)

    # s = cov(x,y) / var(x)
    num = ((x - x_mean) * (y - y_mean)).sum(dim=1, keepdim=True)
    den = ((x - x_mean)**2).sum(dim=1, keepdim=True)
    s = num / (den + 1e-8)
    t = y_mean - s * x_mean

    # Reshape s, t to [B, 1, 1, 1] for broadcasting
    return s.view(B, 1, 1, 1) * depth + t.view(B, 1, 1, 1)

def predict3_batch(depth, lr_depth):
    """
    Median/MAD Alignment in Disparity Space for a Batch.
    """
    B, C, H, W = depth.shape
    max_depth = lr_depth.view(B, -1).max(dim=1)[0].view(B, 1, 1, 1)

    # Inverse depth (Disparity)
    lr_d = 1.0 / (lr_depth + 1e-8)
    disp = torch.where(depth > 1e-3, 1.0 / depth, 1.0 / max_depth)

    # Use median across spatial dimensions [B, N]
    def get_median(t):
        return t.view(B, -1).median(dim=1)[0].view(B, 1, 1, 1)

    t = get_median(lr_d)
    depth_centered = disp - get_median(disp)
    lr_d_centered = lr_d - t

    # Scale using Mean Absolute Deviation
    depth_scale = torch.abs(depth_centered).view(B, -1).mean(dim=1).view(B, 1, 1, 1)
    lr_d_scale = torch.abs(lr_d_centered).view(B, -1).mean(dim=1).view(B, 1, 1, 1)

    s = lr_d_scale / (depth_scale + 1e-8)

    d_metric = (s * depth_centered + t)
    # Clip to avoid negative/zero depth after transformation
    d_metric = torch.clamp(d_metric, min=1/max_depth.max())

    return 1.0 / d_metric

"""### Metrics"""

def eval_depth_batch(pred, target):
    """
    Computes depth metrics for a batch [B, 1, H, W].
    Returns a dictionary of mean metrics across the batch.
    """
    assert pred.shape == target.shape

    # Create mask for valid pixels (where target > 0)
    mask = (target > 0).float()
    n_valid = mask.view(mask.size(0), -1).sum(dim=1) # Valid pixels per image

    # Avoid division by zero for images with no valid pixels
    n_valid_safe = torch.clamp(n_valid, min=1.0)

    def masked_mean(tensor):
        # Sum only valid pixels and divide by valid count per image, then mean over batch
        pixel_sum = (tensor * mask).view(mask.size(0), -1).sum(dim=1)
        return (pixel_sum / n_valid_safe).mean()

    # Threshold metrics (d1, d2, d3)
    # Use 1e-6 to avoid div by zero if pred has 0s
    ratio = torch.max(target / (pred + 1e-6), pred / (target + 1e-6))
    d1 = masked_mean((ratio < 1.25).float())
    d2 = masked_mean((ratio < 1.25**2).float())
    d3 = masked_mean((ratio < 1.25**3).float())

    # Standard error metrics
    diff = pred - target
    abs_rel = masked_mean(torch.abs(diff) / (target + 1e-6))
    rmse = torch.sqrt(((diff**2 * mask).view(mask.size(0), -1).sum(dim=1) / n_valid_safe).mean())
    mae = masked_mean(torch.abs(diff))

    # SILog (Scale Invariant Logarithmic error)
    pred_log = torch.log(torch.clamp(pred, min=1e-3))
    target_log = torch.log(torch.clamp(target, min=1e-3))
    diff_log = (pred_log - target_log) * mask

    # Mean of squared log diff per image
    mean_sq_log = (diff_log**2).view(mask.size(0), -1).sum(dim=1) / n_valid_safe
    # Square of mean log diff per image
    sq_mean_log = (diff_log.view(mask.size(0), -1).sum(dim=1) / n_valid_safe)**2

    silog = torch.sqrt(torch.mean(mean_sq_log - 0.5 * sq_mean_log)) * 100 # Often scaled by 100

    return {
        'd1': d1.item(),
        'd2': d2.item(),
        'd3': d3.item(),
        'abs_rel': abs_rel.item(),
        'rmse': rmse.item(),
        'mae': mae.item(),
        'silog': silog.item()
    }

"""### Validation"""

split='Validation'

# print('Download begin...')

# dataset_downloader = ARKitDataset(split=split, dataset='raw')
# dataset_downloader.all_download()
# dataset_downloader = ARKitDataset(split=split)
# dataset_downloader.all_download()

# print('Download complete')

dataset = ARKitTrainDataset(split=split, transform=transform)
train_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
# batch_rgb, batch_depth, batch_lr_depth, fx_raw = next(iter(train_loader))
# print(f"RGB batch shape: {batch_rgb.shape}")     # Expected: [batch_size, 3, H, W]
# print(f"Depth batch shape: {batch_depth.shape}") # Expected: [batch_size, 1, H, W]
# print(f"Lowres Depth batch shape: {batch_lr_depth.shape}")
# print(f"f_px?: {fx_raw.shape}")

from itertools import islice


methods_names = [ 'raw', 'linalg of depth', 's and t out of disparity']
methods = {'raw' : predict0, 'linalg of depth' : predict1_batch, 's and t out of disparity' : predict3_batch}

stats = {
    m: {
        'd1': [], 'd2': [], 'd3': [],
        'abs_rel': [],
        'rmse': [], 'mae': [], 'silog': []
    } for m in methods_names
}


img_count = 0

for rgb, hr_depth, lr_depth, f_px in islice(train_loader, None):

    input_tensor = rgb.to(DEVICE)
    # lr_depth = lr_depth.to(DEVICE)
    # hr_depth = hr_depth.to(DEVICE)


    if torch.is_tensor(f_px) and f_px.ndim == 1:
        f_px = f_px.view(-1, 1, 1, 1)
    f_px = f_px.to(DEVICE)

    with torch.inference_mode():
        output = model.infer(input_tensor, f_px=f_px)


    depth = output["depth"].detach().to('cpu')
    if depth.ndim == 3: # [B, H, W] -> [B, 1, H, W]
        depth = depth.unsqueeze(1)
    elif depth.ndim == 2: # [H, W] -> [1, 1, H, W]
        depth = depth.unsqueeze(0).unsqueeze(0)


    for m in methods_names:
        # print(depth.shape, lr_depth.shape)
        pred = methods[m](depth, lr_depth)
        # print(pred.shape, hr_depth.shape)
        predictions = eval_depth_batch(pred, hr_depth)
        if predictions is not None:
            for key in predictions:
                # print(m, key, predictions[key], rgb.size(0))
                stats[m][key].append(predictions[key]*rgb.size(0))
        if img_count % 100 == 0:
            print(m, predictions)
        if img_count % 1000 == 0 and show_imgs:
            rgb_to_show = rgb[0].permute(1, 2, 0) #.detach().cpu().numpy()
            show_res(rgb_to_show.detach().numpy(), lr_depth.detach().numpy()[0, 0], hr_depth.detach().numpy()[0, 0].copy(), pred.detach().numpy()[0, 0], name+ m + str(img_count))
            # show_res(rgb_to_show, lr_depth.detach().cpu().numpy()[0, 0], hr_depth.detach().cpu().numpy()[0, 0], pred.detach().cpu().numpy()[0, 0], 'm_' + m + str(img_count))

    del output, pred, input_tensor
    if img_count % 100 == 50:
        gc.collect()
        torch.cuda.empty_cache()


    img_count += rgb.size(0)


final_results = {}
for m in methods_names:
    # Average each metric list into a single value
    # for key, val in stats[m].items():
    #     print(np.sum(val), img_count)
    final_results[m] = {key: np.sum(val)/img_count for key, val in stats[m].items()}

# Convert to DataFrame for a clean, readable table
df = pd.DataFrame(final_results).T
print(df)
df.to_csv(name+"_results_metric.csv", index=False)

detailed_data = []

for method in methods_names:
    for i in range(len(stats[method]['rmse'])):
        row = { 'method': method, 'image_idx': i }
        # Add all metrics for this specific image
        for metric in stats[method]:
            row[metric] = stats[method][metric][i]
        detailed_data.append(row)

# 2. Create DataFrame and Save
df_detailed = pd.DataFrame(detailed_data)
df_detailed.to_csv(name+"_mde_full_results_metric.csv", index=False)
print("Detailed per-image results saved!")

# print(results2)
# print(results3)