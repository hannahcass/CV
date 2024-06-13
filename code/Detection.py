import glob
import json
import os

import cv2
import numpy as np
import torch
from architectures import NewNorm2b
from datasets import DatasetCropped

device = "cpu"
datapath = r"..\my_data\test"
modelpath = r"models\trained_model-NewNorm2b-DatasetRandomCrop-2022-01-21.pt"
outputdir = r"..\detections"

# crop point relative to original dim 1024x1024
# add to final prediction to obtain correct coordinates
X, Y = 108, 334

# get folder names to add as dictionary keys later
img_paths = glob.glob(os.path.join(datapath, '**/*.png'), recursive=True)
img_paths.sort(key=str)
foldernames = []
for path in img_paths:
    foldernames.append(os.path.dirname(path).split('\\')[-1])


dataset = DatasetCropped(datapath)

model = NewNorm2b()
statedict = torch.load(modelpath, map_location=torch.device(device))
model.load_state_dict(statedict)
model.eval()


def diff(inp, outp):
    d = abs(inp - outp)
    return d


def rgb2gray(image, r=0.45, g=0.18, b=0.37):
    gr = image[..., 0] * r + image[..., 1] * g + image[..., 2] * b
    return gr.astype(image.dtype)


def conditions(x, y, w, h):
    minArea = 100
    dim0 = 760
    dim1 = 440
    return x and y and w*h > minArea and x+w<dim0 and y+h<dim1


predictions = {}

# for now for lack of time just discard all temporal info
# take only every 4th image from each folder (timeframe '0')
for i in range(3, len(dataset), 7):
    img = torch.from_numpy(dataset[i])
    img = torch.unsqueeze(img, 0)
    numpy_img=img.numpy()
    #model.eval()
    numpy_output=model(img.to(device)).detach().cpu().numpy()
    numpy_img = np.moveaxis(numpy_img, 1, -1)
    numpy_output = np.moveaxis(numpy_output, 1, -1)
    difference=(diff(numpy_img, numpy_output)*255).astype(np.uint8)
    image = np.squeeze(difference, axis=0)

    gray = rgb2gray(image)
    filtered = cv2.bilateralFilter(gray, d=8, sigmaColor=20, sigmaSpace=60)

    _, image_result = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU*2)

    cnts, _ = cv2.findContours(image_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pred = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if conditions(x, y, w, h):
            pred.append([x+X,y+Y,w,h])

    predictions[foldernames[i]] = pred

with open(os.path.join(outputdir, "predictions.json"), "w") as f:
    json.dump(predictions, f)
    f.close()
