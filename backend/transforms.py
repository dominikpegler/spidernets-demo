# copied from spidernets-analysis/src/spidernets/ on 5 Nov 2025

from torchvision import transforms
from torchvision.transforms import InterpolationMode


def get_data_transforms(input_size, use_bicubic: bool = False):
    """
    Build train/val transforms. For ViT/DeiT/swin/DINOv2, set use_bicubic=True
    to match common pretraining recipes; CNNs can keep bilinear.
    """
    interp = InterpolationMode.BICUBIC if use_bicubic else InterpolationMode.BILINEAR

    return {
        "train": transforms.Compose(
            [
                transforms.Resize([input_size, input_size], interpolation=interp),
                transforms.RandomHorizontalFlip(),
                # transforms.RandomVerticalFlip(),
                transforms.RandomAffine(
                    degrees=1,
                    translate=(0.0, 0.0),
                    fill=[125, 255, 0],
                    shear=(-6, 6, -6, 6),
                    scale=(1.1, 1.1),
                ),
                transforms.ColorJitter(
                    brightness=(0.9, 1.10),
                    contrast=(0.85, 1.15),
                    saturation=(0.95, 1.15),
                ),
                transforms.GaussianBlur(kernel_size=(1, 5), sigma=(0.01, 0.5)),
                transforms.ToTensor(),
                # Keep current ImageNet normalization; you set input_size/model_name manually.
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(input_size, interpolation=interp),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }
