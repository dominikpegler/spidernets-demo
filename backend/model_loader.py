# copied from spidernets-analysis/src/spidernets/ on 5 Nov 2025

import os
from collections import OrderedDict
import requests
from tqdm import tqdm
import torch
from torch import nn
from torchvision import models
import timm


def get_final_layer_name(model_name):
    if model_name.startswith(("resnetv2", "resnet")):
        return "fc"
    if model_name.startswith("mobilenet"):
        return "classifier"
    if model_name.startswith("efficientnet"):
        return "classifier"
    if model_name.startswith(("vit", "deit")):
        return "head"
    return "head.fc"


def set_parameter_requires_grad(model, feature_extracting):
    """
    This helper function sets the ``.requires_grad`` attribute of the
    parameters in the model to False when we are feature extracting.
    """
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = True


class SigmoidScaledLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(SigmoidScaledLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.linear(x)
        x = torch.sigmoid(x) * 100
        return x


# works with only one dimension, but output range of last linear layer is restricted to range of 0-100 using a scaled sigmoid
class SigmoidOutputLayers(nn.Sequential):
    def __init__(self, n_layers, n_ftrs, n_dense, dropout):
        super().__init__(self.init_modules(n_layers, n_ftrs, n_dense, dropout))

    def init_modules(self, n_layers, n_ftrs, n_dense, dropout):
        modules = OrderedDict()

        i = 0
        modules[f"dropout_{i}"] = nn.Dropout(p=dropout)
        modules[f"fc_{i}"] = nn.Linear(n_ftrs, n_dense)

        for i in range(1, n_layers):
            modules[f"relu_{i}"] = nn.ReLU(inplace=True)
            modules[f"batchnorm_{i}"] = nn.BatchNorm1d(n_dense)
            modules[f"dropout_{i}"] = nn.Dropout(p=dropout)
            modules[f"fc_{i}"] = nn.Linear(n_dense, n_dense)

        modules[f"relu_{i+1}"] = nn.ReLU(inplace=True)
        modules[f"batchnorm_{i+1}"] = nn.BatchNorm1d(n_dense)
        modules[f"dropout_{i+1}"] = nn.Dropout(p=dropout)
        modules[f"fc_{i+1}"] = SigmoidScaledLinear(n_dense, 1)

        return modules


# works with more dimensions, but output range of last linear layer is not restricted
class OutputLayers(nn.Sequential):
    def __init__(self, n_layers, n_ftrs, n_dense, dropout, n_dim):
        super().__init__(self.init_modules(n_layers, n_ftrs, n_dense, dropout, n_dim))

    def init_modules(self, n_layers, n_ftrs, n_dense, dropout, n_dim):
        modules = OrderedDict()

        i = 0
        modules[f"dropout_{i}"] = nn.Dropout(p=dropout)
        modules[f"fc_{i}"] = nn.Linear(n_ftrs, n_dense)

        for i in range(1, n_layers):
            modules[f"relu_{i}"] = nn.ReLU(inplace=True)
            modules[f"batchnorm_{i}"] = nn.BatchNorm1d(n_dense)
            modules[f"dropout_{i}"] = nn.Dropout(p=dropout)
            modules[f"fc_{i}"] = nn.Linear(n_dense, n_dense)

        modules[f"relu_{i+1}"] = nn.ReLU(inplace=True)
        modules[f"batchnorm_{i+1}"] = nn.BatchNorm1d(n_dense)
        modules[f"dropout_{i+1}"] = nn.Dropout(p=dropout)
        modules[f"fc_{i+1}"] = nn.Linear(n_dense, n_dim)

        return modules


def create_model(
    run_config,
    n_layers,
    dropout,
    # data_dir,
    n_dense=256,
    feature_extraction=True,
):
    """
    Create an initial model based on the specified architecture.

    Args:
        model_name (str): The name of the model to use. Can be "resnet50", "resnet152",
        or any model name compatible with the TIMM library.
        target_vars (list): The target variables for the model to predict.
        n_layers (int): Number of layers in the output section of the model.
        dropout (float): Dropout rate to apply in the output section of the model.
        n_dense (int, optional): Number of dense units in the output layers. Default is 256.
        feature_extraction (bool, optional): Whether to set the model to feature extraction
        mode (True) or not (False). Default is True.

    Returns:
        nn.Module: A PyTorch neural network model with the specified architecture and adjustments.

    Example:
        model = create_initial_model("resnet50", ["label"], 3, 0.5)
    """

    final_layer = get_final_layer_name(run_config["model_name"])

    # torchvision models
    if run_config["model_name"] == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    elif run_config["model_name"] == "resnet152":
        model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
    elif run_config["model_name"] == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif run_config["model_name"] == "efficientnet_b0":
        model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
    elif run_config["model_name"] == "efficientnet_b1":
        model = models.efficientnet_b1(
            weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1
        )
    elif run_config["model_name"] == "efficientnet_b2":
        model = models.efficientnet_b2(
            weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1
        )
    elif run_config["model_name"] == "efficientnet_b3":
        model = models.efficientnet_b3(
            weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1
        )
    elif run_config["model_name"] == "efficientnet_b4":
        model = models.efficientnet_b4(
            weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1
        )
    elif run_config["model_name"] == "efficientnet_b5":
        model = models.efficientnet_b5(
            weights=models.EfficientNet_B5_Weights.IMAGENET1K_V1
        )
    elif run_config["model_name"] == "efficientnet_v2_s":
        model = models.efficientnet_v2_s(
            weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        )
    # elif run_config["model_name"] == "resnet50_in21k":
    #     model = load_resnet50_imagenet21k(data_dir)  # custom weights
    # search model on TIMM
    else:
        is_vit_like = run_config["model_name"].lower().startswith(("vit", "deit")) or (
            "dinov2" in run_config["model_name"].lower()
        )
        kwargs = dict(pretrained=True, num_classes=0)
        if is_vit_like and run_config.get("input_size"):
            kwargs["img_size"] = int(run_config["input_size"])
        model = timm.create_model(run_config["model_name"], **kwargs)

    set_parameter_requires_grad(model, feature_extraction)

    if hasattr(model, "num_features"):  # timm models
        n_ftrs = model.num_features
    else:
        # torchvision fallback (incl. efficientnet special-case)
        if run_config["model_name"] in [
            "efficientnet_b0",
            "efficientnet_b1",
            "efficientnet_b2",
            "efficientnet_b3",
            "efficientnet_b4",
            "efficientnet_b5",
            "efficientnet_v2_s",
        ]:
            n_ftrs = model.get_submodule(final_layer)[1].in_features
        else:
            n_ftrs = model.get_submodule(final_layer).in_features

    n_dim = len(run_config["target_vars"])

    if run_config["final_activation"] == "sigmoid":
        if n_dim > 1:
            raise NotImplementedError(
                "SigmoidOutputLayers supports only 1 output dimension."
            )
        new_layers = SigmoidOutputLayers(n_layers, n_ftrs, n_dense, dropout)
    elif run_config["final_activation"] == "linear":
        new_layers = OutputLayers(n_layers, n_ftrs, n_dense, dropout, n_dim)
    else:
        raise ValueError("Final activation must be 'sigmoid' or 'linear'.")

    if "." in final_layer:
        outer, inner = final_layer.split(".")
        parent = model.get_submodule(outer)
        # If parent is Identity or doesn't have inner, replace whole parent
        if isinstance(parent, nn.Identity) or not hasattr(parent, inner):
            setattr(model, outer, new_layers)
        else:
            parent.add_module(inner, new_layers)
    else:
        setattr(model, final_layer, new_layers)

    return model


def download_file(url, dest, timeout=30):
    """
    Downloads a file from a URL to a destination path.
    """
    response = requests.get(url, stream=True, timeout=timeout)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024
    t = tqdm(total=total_size, unit="iB", unit_scale=True)

    with open(dest, "wb") as file:
        for data in response.iter_content(block_size):
            t.update(len(data))
            file.write(data)
    t.close()
    if total_size != 0 and t.n != total_size:
        raise ValueError(
            "Error during download, downloaded file size does not match expected size."
        )


def load_resnet50_imagenet21k(data_dir):
    """
    Load ResNet50 model trained on the ImageNet21k dataset.

    Source: https://github.com/Alibaba-MIIL/ImageNet21K
    """
    checkpoint_dir = os.path.join(data_dir, "checkpoint_resnet50_in21k")
    checkpoint_path = os.path.join(checkpoint_dir, "resnet50_miil_21k.pth")
    checkpoint_url = "https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/resnet50_miil_21k.pth"

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Downloading checkpoint from {checkpoint_url} to {checkpoint_path}")
        try:
            download_file(checkpoint_url, checkpoint_path, timeout=30)
        except ConnectionError as e:
            print(f"Failed to download the checkpoint: {e}")
            return None

    checkpoint = torch.load(checkpoint_path)
    model = models.resnet50()
    n_classes = checkpoint["num_classes"]
    n_ftrs = model.fc.in_features

    # Replace final layer
    model.fc = nn.Linear(n_ftrs, n_classes)
    model.load_state_dict(checkpoint["state_dict"])

    return model
