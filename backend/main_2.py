import io
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
from torchvision import transforms

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["dominikpegler.github.io"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# Load model once at startup
model = torch.jit.load("model.pt")  # or torch.load(...)
model.eval()
preproc = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(), transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    x = preproc(img).unsqueeze(0)
    with torch.inference_mode():
        y = model(x).item()  # adapt to your head
    return {"prediction": y}