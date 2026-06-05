from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
import numpy as np
import cv2
from model import UNet
import io

app = FastAPI()

DEVICE = "cpu"
THRESHOLD = 0.5
MODEL_PATH = "unet_tgs_salt.pth"

model = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model.eval()

@app.get("/")
def root():
    return {"message": "U-Net Segmentation API çalışıyor!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype("float32") / 255.0
    image = cv2.resize(image, (128, 128))

    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, 0)
    image = torch.from_numpy(image).to(DEVICE)

    with torch.no_grad():
        predMask = model(image).squeeze()
        predMask = torch.sigmoid(predMask)
        predMask = predMask.cpu().numpy()

    predMask = (predMask > THRESHOLD) * 255
    predMask = predMask.astype("uint8")

    saltPixels = int((predMask == 255).sum())
    totalPixels = int(predMask.size)
    saltRatio = round(saltPixels / totalPixels, 4)
    label = "salt" if saltRatio > THRESHOLD else "no salt"

    return JSONResponse(content={
        "label": label,
        "salt_ratio": saltRatio,
        "salt_pixels": saltPixels,
        "total_pixels": totalPixels
    })