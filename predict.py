'''
This is the code used to calculate the viabilities and other metrics about the spheroids in images that you upload
This code will work for images of cystic spheroids
If you are trying to use other types of images, please train the model according to the READ ME instructions
    and run the training.py code to train a custom model for your images/data.
'''

# importing all relevant packages
# make a note of which packages are used for what and/or note which groupings of pacakges assist with the tasks in this code - KT

import typing
from typing import Tuple, List, Dict, Union, Any, Optional, Iterable

import torch
import os
import time
from tqdm import tqdm # ??? why not just import tqdm  - KT
import pandas as pd
import argparse

from PIL import Image # why are we using the deprecated version of pillow? - KT

import matplotlib.pyplot as plt
import torchvision.transforms as T
import torchvision
import numpy as np
import cv2
import random
import warnings

# adding input arguments that the user must indicate when running this code
parser = argparse.ArgumentParser()
parser.add_argument("--input", help="folder of images to analyze")
parser.add_argument("--output", help="folder to save output images")
parser.add_argument("--model", help="path to model")

# setting the input arguments to a variable? is this correct? - KT
args = parser.parse_args()
if args.input is None or args.output is None:
    print("please provide input and output folders")
    exit()

# checking if the device being used will run through gpu or cpu
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print("Loading model...")

if device == torch.device("cpu"):
    model = torch.load(args.model, map_location=torch.device("cpu"))
else:
    model = torch.load(args.model)
# model = torch.load("debug01.pt") --> clean up the code if this is not needed - KT
print("Model loaded.")
print("running on device: ", device)
model.to(device)

model.eval() # begins the image analysis
CLASS_NAMES = ["__background__", "cell"] #  these class names are defined in training.py? - KT
warnings.filterwarnings("ignore") # why are we ignoring warnings? what does this do? - KT

# analyzing the input images and calculating the necessary information from the identified regions within the image - KT
def get_prediction(img_path, confidence):
    img = Image.open(img_path).convert("RGB")  # get rid of alpha channel
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    img = img.to(device)
    pred = model([img])
    pred_score = list(pred[0]["scores"].detach().cpu().numpy())
    filtered_pred_indices = [pred_score.index(x) for x in pred_score if x > confidence]

    # what does this do? - KT
    if not filtered_pred_indices:
        return [], [], [], None

    # what is this for? - KT
    pred_t = filtered_pred_indices[-1]  # it is the index of the last prediction that has a score higher than the confidence threshold

    # obtaining masks from the input images - KT
    masks = (pred[0]["masks"] > 0.5).squeeze().detach().cpu().numpy()
    # print(pred[0]['labels'].numpy().max()) --> if you don't need this, clease this up. If you do, note what it can be used for - KT
    pred_class = [CLASS_NAMES[i] for i in list(pred[0]["labels"].cpu().numpy())]
    
    # note what this does - KT
    pred_boxes = [
        [(i[0], i[1]), (i[2], i[3])]
        for i in list(pred[0]["boxes"].detach().cpu().numpy())
    ]
    masks = masks[: pred_t + 1]
    pred_boxes = pred_boxes[: pred_t + 1]
    pred_class = pred_class[: pred_t + 1]
    confidence_scores = pred_score[: pred_t + 1]
    return masks, pred_boxes, pred_class, confidence_scores

# segmenting each identified instance on the input images - KT
# does this also print the identification onto the output images with the itendified regions? - KT
# summarize what the following function does - KT 
def segment_instance(img_path: str, confidence_thresh=0.5, rect_th=2, text_size=2, text_th=2) -> tuple[Any, Any]:
    
    masks, boxes, pred_cls, confidence_scores = get_prediction(img_path, confidence_thresh)

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # initializing the classes? - KT
    cells = [] # after reading through the rest of the code, this seems to the the initialization of a list containing all the calculated information about each region in each image. If this is the case, I would rename this to make it more clear, since it is currently named after one of the classes and you want to make sure that this is 
    backgroundIntesity = calcBackgroundIntensity(img, masks)

    # assigning information to the identified regions as masks - KT
    for i in range(len(masks)):
        pt1 = tuple(map(int, boxes[i][0]))
        pt2 = tuple(map(int, boxes[i][1]))

        x, y = pt2

        contours, _ = cv2.findContours(
            masks[i].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if len(contours) == 0:
            continue
        (x, y), (minorAxisLength, majorAxisLength), angle = cv2.fitEllipse(
            max(contours, key=cv2.contourArea)
        )
        perimeter = 0
        for contour in contours:
            perimeter += cv2.arcLength(contour, True)
        
        # Draw the contours on the image
        cv2.drawContours(img, contours, -1, (0, 255, 0), rect_th)

        # get the average intensity of all pixels within mask
        imgSave = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgSave = imgSave * masks[i]

        viability, averageIntensity, area = analyzeCell(imgSave, backgroundIntesity)

        # what is the check that you are introducing here? - KT
        if area == 0:
            continue

        # the following groupings print information identified and calculated about each segmented region onto the output image - KT
        cv2.putText(
            img,
            str(round(viability, 2)),
            (int(x + 10), int(y + 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1,
        )
        cv2.putText(
            img,
            str(confidence_scores[i]),
            (int(x + 10), int(y + 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1,
        )
        masked_image_out = cv2.putText(
            img,
            str(round(averageIntensity, 2)),
            (int(x + 20), int(y + 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1,
        )

        # calculating the circulrity for each identified region - KT 
        circularity = (4 * np.pi * area) / (perimeter ** 2)

        # putting all of the calulcated region data together in one dictionary - KT
        cell_meta = {
            "viability": viability,
            "circularity": circularity,
            "averageIntensity": averageIntensity,
            "area": area,
            "perimeter": perimeter,
        }

        # append the calculated region data to the cells list (defined earlier) - KT
        cells.append(cell_meta)
    return img, cells, backgroundIntesity

# defining a function that will caluclate information about the background (not segmentted regions)
# for use in comparing the background to the segmented regions when calculating viability - KT
def calcBackgroundIntensity(img, masks) -> float:

    """
    Required inputs for this function - KT
    :param img: image
    :param mask: a list of masks
    :return: average intensity of background
    """

    imgSave = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    combined_mask = np.zeros_like(imgSave)

    for mask in masks:
        combined_mask = np.logical_or(combined_mask, mask)

    backgroundMask = np.logical_not(combined_mask)

    imgSave = imgSave * backgroundMask

    # remove the following commented lines if you do not need these - KT
    # plt.imshow(imgSave)
    # plt.show()
    # flatten 2d array to 1d
    # plt.imshow(imgSave, cmap="gray")
    # plt.show()
    # if you use the above, make a note about what these should/could be uncommented for - KT
    imgSave = imgSave.flatten()

    masked = imgSave[imgSave > 5]  # ignore the completely black background
    masked = masked[masked != 255]  # ignore the scale bar white
    # ignore everything greater than 250 --> you don't have code that does this? Either introduce the code or remove it - KT

    avg = np.average(masked)

    return avg

# defining a function where we analyze the segmented regions of interest for viability, etc. calculations
def analyzeCell(cell, backgroundIntensity):
    area = np.count_nonzero(cell)
    cell = cell[cell != 0]  # ignore the completely black background
    averageIntensity = np.average(cell)
    cell_state = (
        (60 - np.clip((backgroundIntensity - 15 - averageIntensity), 0, 60)) / 60
    ) * 100

    # circularity = 0 --> don't we already calculate the circularity above? Why do we need this here? - KT

    return cell_state, averageIntensity, area

#  identify the folder where the input arguments are nested - KT (review this!)
folder = args.input

# identify the images in the input folder for analysis (review) - KT
files = os.listdir(folder)

timeStart = time.time()
# for file in files: --> clean up this code

# empty list that holds all the images (review) - KT
images = []
# make a new list to hold all of the information about the images - KT
images_meta = []

# note what this chunk of code does - KT
for file in tqdm(files):
    if not (file.endswith(".jpg") or file.endswith(".png") or file.endswith(".tiff") or file.endswith(".tif")):
        continue
    
    # check image resolution
    img = cv2.imread(os.path.join(folder,file))

    # Resize image, if appropriate, to max 1290x1290 mataining aspect ratio
    if img.shape[0] > 1290 or img.shape[1] > 1290:

        scale = min(1290/img.shape[0], 1290/img.shape[1])
        img = cv2.resize(img, (0,0), fx=scale, fy=scale)
        # write it into temp directory
        print("resizing image: ", file)

        if not os.path.exists(os.path.join(folder, "temp")):
            os.mkdir(os.path.join(folder, "temp"))

        cv2.imwrite(os.path.join(folder, "temp", file), img)
        file = os.path.join("temp",file)
        # continue

    # what does this code do? - KT
    image, cells, backgroundIntensity = segment_instance(
        os.path.join(folder,file), confidence_thresh=0.8
    )

    image_total_px = image.shape[0] * image.shape[1] # calculate the total pixel area of the image - KT
    sum_area = sum([cell["area"] for cell in cells]) # calculate the total area of segmented regions analyzed - KT
    pct_area = sum_area / image_total_px # calculate the percent area of the image that was analyzed - KT

    # appending analyzed/calculated information from the images
    images_meta.append(
        {
            "file": file,
            "cells": cells,
            "backgroundIntensity": backgroundIntensity,
            "pct_area_analyzed": pct_area,
        }
    )
    images.append(image)

timeEnd = time.time() # note the end time of the function running to get the total time per image analyzed 
print("Time taken: ", timeEnd - timeStart)
print("Time taken per image: ", (timeEnd - timeStart) / len(files))

# exiting the analysis cycle of the code functionality - KT
print("Removing temp files...")

# writing temp stored information analyzed from the images into a csv - KT (review)
if os.path.exists(os.path.join(folder, "temp")):
    for file in os.listdir(os.path.join(folder, "temp")):
        os.remove(os.path.join(folder, "temp", file))
    os.rmdir(os.path.join(folder, "temp"))

# path = os.path.join(folder, "out") --> clean code or note the instance where this could be needed - KT
path = args.output
if not os.path.exists(path):
    os.mkdir(path)

for i in range(len(images)):
    try:
        cv2.imwrite(os.path.join(path, files[i]), images[i])
    except:
        print("error saving image: ", files[i])

# make a new dataframe with empty everything
df = pd.DataFrame()

# storing information in the dataframe for export
for image in images_meta:

    # in the case where NO spheroid is identified in an image 
    avg_area = -1
    avg_perimeter = -1
    avg_circularity = -1
    avg_intensity = -1
    avg_viability = -1
    num_cells = 0
    if image["cells"] != []:
    
        num_cells = len(image["cells"])
        
        avg_viability = np.average(
            [cell["viability"] for cell in image["cells"]]
        ).round(2)
        avg_circularity = np.average(
            [cell["circularity"] for cell in image["cells"]]
        ).round(5)
        avg_intensity = np.average(
            [cell["averageIntensity"] for cell in image["cells"]]
        ).round(2)
        avg_area = np.average([cell["area"] for cell in image["cells"]]).round(2)
        avg_perimeter = np.average([cell["perimeter"] for cell in image["cells"]]).round(2)


    df = pd.concat([df,
        pd.DataFrame(
            {
                # or rearrange the columns here like where I commented above with *** - KT
                "file": [image["file"]],
                "count": [num_cells],
                "pct_analyzed": [image["pct_area_analyzed"] * 100],
                "background_intenstiy": [image["backgroundIntensity"]],
                "avg_viability": [avg_viability],
                "avg_circularity": [avg_circularity],
                "avg_intensity": [avg_intensity],
                "avg_area": [avg_area],
                "avg_perimeter": [avg_perimeter],
            }
        )],
        
        ignore_index=True,
    )

# what does this code do? - KT
try:
    df.to_csv(os.path.join(path, "summary.csv"), index=False)

except PermissionError:
    print("Please close the summary.csv file. press any key to continue")
    input()
    # df.to_csv("out\\summary.csv", index=False)
