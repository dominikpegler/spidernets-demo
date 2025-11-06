# copied from spidernets-analysis/src/spidernets/ on 5 Nov 2025

import os
import pandas as pd
import math
import numpy as np
from skimage.morphology import binary_dilation, disk
import torch
from typing import List, Tuple, Optional, Callable
from pytorch_grad_cam.utils.reshape_transforms import vit_reshape_transform


def get_ensemble_attributions(runs, exp_dir, stage, interpolate_mode, method="gradcam"):
    method_string = "_" + method
    df = pd.DataFrame()
    for run_id in runs:
        run_data_dir = os.path.join(exp_dir, run_id, "data")
        df_run = pd.read_pickle(
            os.path.join(
                run_data_dir,
                f"attributions{method_string}_{stage}_{interpolate_mode}.pkl",
            )
        )
        df_run["run"] = run_id
        df = pd.concat([df, df_run], sort=False)

    assert len(df["true"].unique()) <= len(
        df["img_path"].unique()
    ), "Real values should for all images be the same regardless of run / ensemble model!"

    df = df.groupby("img_path", as_index=False).agg(
        true=("true", "mean"),
        pred_mean=("pred", "mean"),
        pred_std=("pred", "std"),
        attributions_mean=("attributions", "mean"),
    )

    return df


def cohen_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(
        ((nx - 1) * np.std(x, ddof=1) ** 2 + (ny - 1) * np.std(y, ddof=1) ** 2) / dof
    )


def get_key_indices(arr, num_indices, add_stochasticity=False, random_seed=None):
    """
    Finds the indexes of the minimum, maximum, and a specified number of
    evenly spaced intermediate values from an array, with an option to add
    stochasticity.

    Args:
        arr (list or np.ndarray): A list or NumPy array of numerical values.
        num_indices (int): The total number of indexes to return.
                           Must be at least 2.
        add_stochasticity (bool): If True, adds random noise to the steps
                                  for intermediate value selection.
        random_seed (int, optional): A seed for the random number generator
                                     for reproducibility.

    Returns:
        list: A list of integers representing the requested indexes from the
              original array.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    if not isinstance(arr, (list, np.ndarray)) or len(arr) == 0:
        print("Error: Input array is not a valid list or NumPy array or is empty.")
        return []

    if not isinstance(num_indices, int) or num_indices < 2:
        print("Error: num_indices must be an integer of 2 or more.")
        return []

    if num_indices >= len(arr):
        return list(range(len(arr)))

    indices = list(range(len(arr)))
    indices.sort(key=lambda i: arr[i])

    min_index = indices[0]
    max_index = indices[-1]

    intermediate_indices = indices[1:-1]
    num_intermediate = num_indices - 2

    if num_intermediate <= 0:
        return sorted([min_index, max_index], key=lambda i: arr[i])

    spacing = len(intermediate_indices) / (num_intermediate + 1)

    result_indices = [min_index]

    for i in range(num_intermediate):
        # Add a normally distributed random value to the spacing if stochasticity is enabled.
        current_spacing = spacing
        if add_stochasticity:
            noise = np.random.normal(0, 0.01)
            current_spacing += noise

        position = math.floor(current_spacing * (i + 1))

        # Ensure the position stays within the valid bounds of intermediate_indices
        position = max(0, min(position, len(intermediate_indices) - 1))

        # Append a new intermediate index to the result list.
        result_indices.append(intermediate_indices[position])

    result_indices.append(max_index)

    return sorted(list(set(result_indices)), key=lambda i: arr[i])


def thicken_mask(mask_array: np.ndarray, final_linewidth_px: int) -> np.ndarray:
    """Thickens an image mask.

    Args:
        mask_array (np.ndarray): 2D array of 0s/1s.
        final_linewidth_px (int): Final thickness in pixels.

    Returns:
        np.ndarray: The thickened mask.
    """
    boolean_mask = mask_array.astype(bool)
    # Convert the radius to an integer to avoid a tuple index error.
    structuring_element = disk(int(final_linewidth_px / 2))
    thickened_mask = binary_dilation(boolean_mask, footprint=structuring_element)
    return thickened_mask.astype(int)


# -----------------------------
# New helpers for pytorch-grad-cam
# -----------------------------


class RegressionOutputTarget:
    """Callable target for regression outputs."""

    def __init__(self, index: int = 0):
        self.index = index

    def __call__(self, y: torch.Tensor):
        if y.ndim == 0:
            return y
        if y.ndim == 1:
            return y
        if y.shape[1] == 1:
            return y.squeeze(1)
        return y[:, self.index]


def make_swin_reshape_transform(
    model: torch.nn.Module,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Swin: map activations to [B, C, H, W]. Handles both [B,H,W,C] and [B,N,C]."""
    blk = model.layers[-1].blocks[-1]
    H, W = blk.input_resolution

    def _transform(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 4:
            # [B, H, W, C] -> [B, C, H, W]
            return tensor.permute(0, 3, 1, 2).contiguous()
        # else assume [B, N, C]
        B, N, C = tensor.shape
        x = tensor.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return x

    return _transform


def make_vit_reshape_transform(model: torch.nn.Module) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    ViT/DeiT/DINOv2: map [B, N, C] -> [B, C, H, W], dropping all prefix tokens (cls/dist/register).
    """
    # Grid from the patch embed (matches img_size used to build the model)
    H, W = model.patch_embed.grid_size
    num_patches = int(H * W)

    def _transform(t: torch.Tensor) -> torch.Tensor:
        # If already [B, H, W, C], just permute
        if t.ndim == 4:
            return t.permute(0, 3, 1, 2).contiguous()

        # Expected [B, N, C]
        B, N, C = t.shape

        # Number of prefix tokens (cls/dist/register, etc.)
        n_prefix = N - num_patches
        if n_prefix < 0:
            # Fallback: infer a square grid from N if needed
            # (should not happen if img_size was set consistently)
            num_p = N
            side = int(round(num_p ** 0.5))
            x = t.reshape(B, side, side, C).permute(0, 3, 1, 2).contiguous()
            return x

        x = t[:, n_prefix:, :]  # drop all prefix tokens
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return x

    return _transform

def pick_cam_targets(model: torch.nn.Module, model_name: str):
    name = (model_name or "").lower()
    if name.startswith("resnet"):
        return [model.layer4[-1].conv3], None
    if name.startswith("convnextv2"):
        blk = model.stages[-1].blocks[-1]
        target = getattr(blk, "dwconv", getattr(blk, "conv_dw", blk))
        return [target], None
    if name.startswith("swin"):
        return [model.layers[-1].blocks[-1].norm1], make_swin_reshape_transform(model)
    if name.startswith(("vit", "deit")) or ("dinov2" in name):
        # Keep using a late transformer block norm for CAM
        return [model.blocks[-1].norm1], make_vit_reshape_transform(model)
    raise NotImplementedError(f"No CAM targets defined for model '{model_name}'")



def gradcam_maps(
    model: torch.nn.Module,
    model_name: str,
    input_tensor: torch.Tensor,
    target_index: int = 0,
) -> np.ndarray:
    target_layers, reshape_transform = pick_cam_targets(model, model_name)
    cam = GradCAM(
        model=model, target_layers=target_layers, reshape_transform=reshape_transform
    )
    targets = [RegressionOutputTarget(target_index)]
    maps = cam(input_tensor=input_tensor, targets=targets)  # numpy [B, H, W]
    return maps


def create_split_bins(min_val, max_val, nbins):
    if min_val >= 0:
        return np.linspace(min_val, max_val, nbins + 1)
    if max_val <= 0:
        return np.linspace(min_val, max_val, nbins + 1)
    range_total = max_val - min_val
    n_neg = int(np.round((abs(min_val) / range_total) * nbins))
    n_pos = nbins - n_neg

    if n_neg == 0 and abs(min_val) > 1e-9:
        n_neg = 1
        n_pos = max(1, nbins - 1)
    if n_pos == 0 and abs(max_val) > 1e-9:
        n_pos = 1
        n_neg = max(1, nbins - 1)
    bins_neg = np.linspace(min_val, 0, n_neg + 1)[:-1]
    bins_pos = np.linspace(0, max_val, n_pos + 1)
    return np.concatenate((bins_neg, bins_pos))