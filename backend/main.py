"""
FastAPI backend.

SQLAlchemy tables are defined here, not in a separate
file. This might change in future versions.
"""

import logging
import pickle
import io
from pathlib import Path
import json
import torch
import numpy as np
import cv2
import base64
from PIL import Image
from transforms import get_data_transforms

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pytorch_grad_cam import GradCAM
from gradcam import (
    pick_cam_targets,
    RegressionOutputTarget,
)
from model_loader import create_model


img_path = Path("checkpoint", "Sp_004.jpg")

is_transformer = True
device = "cpu"

transform = get_data_transforms(224, use_bicubic=is_transformer)["val"]

with open(Path("checkpoint", "run_config.json")) as fp:
    run_config = json.load(fp)

params = {}
with open(
    Path(
        "checkpoint",
        f"params_feature_extraction.json",
    ),
    encoding="utf-8",
) as fp:
    params["feature_extraction"] = json.load(fp)
with open(
    Path(
        "checkpoint",
        f"params_fine_tuning.json",
    ),
    encoding="utf-8",
) as fp:
    params["fine_tuning"] = json.load(fp)

m = create_model(
    run_config,
    n_layers=params["feature_extraction"]["n_layers"],
    dropout=params["feature_extraction"]["dropout"],
    n_dense=256,
    feature_extraction=False,
)


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)


checkpoint_path = Path("checkpoint", "checkpoint_fine_tuning.pth")

with open(checkpoint_path, "rb") as fp:
    checkpoint = CPU_Unpickler(fp).load()

m.load_state_dict(checkpoint)

m.eval()

##############
# LOGGING    #
##############


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create a file handler and set its level to DEBUG
file_handler = logging.FileHandler(filename="server.log")
file_handler.setLevel(logging.DEBUG)

# Create a stream handler and set its level to INFO
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)

# Create a formatter and add it to the handlers
formatter = logging.Formatter("%(levelname)s:    %(message)s")
formatter_file = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter_file)
stream_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

##############
# APP CONFIG #
##############


app = FastAPI(
    docs_url=None,  # Disable docs (Swagger UI)
    redoc_url=None,  # Disable redoc
)

origins = ["http://localhost", "http://localhost:8080", "http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


#############
# API CALLS #
#############


# test post requests: curl -X POST "http://localhost:8000/xid/" \\
# -H "Content-Type: application/json" -d '{"pid": "Hello"}'


@app.post("/post-img/")
async def post_img(image: UploadFile = File(...)):
    """
    Receive an image via multipart/form-data (field name 'image'),
    preprocess, run the model, and return a float in JSON.
    """
    contents = await image.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img_t = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = m(img_t)
        output_np = output.cpu().numpy().squeeze()

    fear = float(np.clip(output_np, 0.0, 100.0))

    target_layers, reshape_tf = pick_cam_targets(m, run_config["model_name"])
    cam = GradCAM(
        model=m,
        target_layers=target_layers,
        reshape_transform=reshape_tf,
    )
    cam_target = [RegressionOutputTarget(0)]

    # Grad-CAM map (returned as numpy, HxW in input resolution)
    grayscale_cam = cam(input_tensor=img_t, targets=cam_target)[0]  # (H, W), [0,1]

    cam_resized = cv2.resize(
        grayscale_cam,
        (img.width, img.height),
        interpolation=cv2.INTER_LINEAR,
    )

    arr = cam_resized.astype(np.float32, copy=False)
    h, w = arr.shape
    b64 = base64.b64encode(arr.tobytes(order="C")).decode("ascii")
    payload = {
        "value": float(fear),
        "array_b64": b64,
        "shape": [int(h), int(w)],
        "dtype": "float32",
    }
    return JSONResponse(payload)


# DOWNLOAD/TEST CALLS


@app.get("/img-id/{img_id}")
def test_img_id(img_id: int):

    return img_id


@app.get("/test/")
def test():
    return {"I am": "alive"}
