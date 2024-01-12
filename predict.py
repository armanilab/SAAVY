'''
This is the code used to calculate the viabilities and other metrics about the spheroids in images that you upload
This code will work for images of cystic spheroids
If you are trying to use other types of images, please train the model according to the READ ME instructions
    and run the training.py code to train a custom model for your images/data.
'''


import typing
from typing import Tuple, List, Dict, Union, Any, Optional, Iterable
import torch
import os
import time
from tqdm import tqdm 
import pandas as pd
import argparse
from PIL import Image 
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

# parse and verify arguements
args = parser.parse_args()
if args.input is None or args.output is None:
    print("please provide input and output folders")
    exit()

# We will sue CPU if cuda isnt avaliable
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print("Loading model...")

if device == torch.device("cpu"):
    model = torch.load(args.model, map_location=torch.device("cpu"))
else:
    model = torch.load(args.model)
print("Model loaded.")
print("running on device: ", device)
model.to(device)

model.eval() #sets the model into evaluation mode which prevents changing weights 
CLASS_NAMES = ["__background__", "cell"] #  two class type, cell, no cell (background)
warnings.filterwarnings("ignore") 


def get_prediction(img_path, confidence):
    """
    Params: image, a filesystem path that points to an image, 
            confidence, cutoff threshold for what is allowed as cell or not
    Returns: model output containing masks for image
    """
    img = Image.open(img_path).convert("RGB")  # get rid of alpha channel incase image is png
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    img = img.to(device) #send image to either GPU or CPU as decided earlier
    pred = model([img]) # evaluate the image tensor using model
    pred_score = list(pred[0]["scores"].detach().cpu().numpy())
    filtered_pred_indices = [pred_score.index(x) for x in pred_score if x > confidence]

    # if there are no cells detected, return the empty cet
    if not filtered_pred_indices:
        return [], [], [], None

    
    pred_t = filtered_pred_indices[-1]  # it is the index of the last prediction that has a score higher than the confidence threshold

    # get masks from model results and load back into cpu so we can work with it
    masks = (pred[0]["masks"] > 0.5).squeeze().detach().cpu().numpy()
    pred_class = [CLASS_NAMES[i] for i in list(pred[0]["labels"].cpu().numpy())]
    
    # bounding boxes
    pred_boxes = [
        [(i[0], i[1]), (i[2], i[3])]
        for i in list(pred[0]["boxes"].detach().cpu().numpy())
    ]
    masks = masks[: pred_t + 1]
    pred_boxes = pred_boxes[: pred_t + 1]
    pred_class = pred_class[: pred_t + 1]
    confidence_scores = pred_score[: pred_t + 1]
    return masks, pred_boxes, pred_class, confidence_scores


def segment_instance(img_path: str, confidence_thresh=0.5, rect_th=2, text_size=2, text_th=2) -> tuple[Any, Any]:
    """
        :param image path: path to an image
        :param confidence_thresh: cutoff confidence threshold
        :param ret_th: line thickness of output images mask
        :param text_size: text font
        :param text_th: text thickness
        Takes an image and runs primary analysis
        :return img: image with cells traced
        :return cells: dictionary with cell metdata
        :return background_intensity: avergage background intensity (non cell)

    """
    
    masks, boxes, pred_cls, confidence_scores = get_prediction(img_path, confidence_thresh)

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # init cell metadata field, we will write to this for every cell we analyze

    cells = [] 
    backgroundIntesity = calcBackgroundIntensity(img, masks)

    # Iterate over every mask
    for i in range(len(masks)):
        pt1 = tuple(map(int, boxes[i][0]))
        pt2 = tuple(map(int, boxes[i][1]))
        x2, y2 = pt1
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

        # Nonzero area isnt a cell so we can ignore it
        if area == 0:
            continue

        # write the cell information to the input images
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
        cv2.putText(
            img,
            str(i),
            (int(x2+10), int(y2+10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (20, 20, 20),
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

        # calculating the circulrity for each identified cell
        circularity = (4 * np.pi * area) / (perimeter ** 2)


        cell_meta = {
            "id": i,
            "viability": viability,
            "circularity": circularity,
            "averageIntensity": averageIntensity,
            "area": area,
            "perimeter": perimeter,
        }

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


    imgSave = imgSave.flatten()

    masked = imgSave[imgSave > 5]  # ignore the completely black background
    masked = masked[masked != 255]  # ignore the scale bar white

    avg = np.average(masked)

    return avg

def analyzeCell(cell, backgroundIntensity):
    area = np.count_nonzero(cell)
    cell = cell[cell != 0]  # ignore the completely black background
    averageIntensity = np.average(cell)
    cell_state = (
        (60 - np.clip((backgroundIntensity - 15 - averageIntensity), 0, 60)) / 60
    ) * 100

    return cell_state, averageIntensity, area

# main driver code
if __name__ == "__main__":
    folder = args.input

    files = os.listdir(folder)

    timeStart = time.time()

    images = [] # images with cells highlighted
    images_meta = [] # cell metadata such as average viability

    # note what this chunk of code does - KT
    for file in tqdm(files):
        if not (file.endswith(".jpg") or file.endswith(".png") or file.endswith(".tiff") or file.endswith(".tif")):
            continue
        
        # read in the image
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

        # analyze a single image
        image, cells, backgroundIntensity = segment_instance(
            os.path.join(folder,file), confidence_thresh=0.8
        )

        image_total_px = image.shape[0] * image.shape[1] # calculate the total pixel area of the image 
        sum_area = sum([cell["area"] for cell in cells]) # calculate the total area of segmented regions analyzed 
        pct_area = sum_area / image_total_px # calculate the percent area of the image that was analyzed 

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

    #cleanup our resized images and write out output files
    print("Removing temp files...")

    if os.path.exists(os.path.join(folder, "temp")):
        for file in os.listdir(os.path.join(folder, "temp")):
            os.remove(os.path.join(folder, "temp", file))
        os.rmdir(os.path.join(folder, "temp"))

    path = args.output
    if not os.path.exists(path):
        os.mkdir(path)

    for i in range(len(images)):
        try:
            cv2.imwrite(os.path.join(path, files[i]), images[i])
        except:
            print("error saving image: ", files[i])

    # make a new dataframe with empty everything so we can save to CSV
    df = pd.DataFrame() # image level dataframe
    cellDFs = []
    
    for image in images_meta:
        cellDF = pd.DataFrame() # cell level dataframe
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
            for cell in image["cells"]:
                cellDF = pd.concat([cellDF,
                                    pd.DataFrame(
                    {
                        "id": [cell["id"]],
                        "viability": [cell["viability"]],
                        "circularity": [cell["circularity"]],
                        "averageIntensity": [cell["averageIntensity"]],
                        "area": [cell["area"]],
                        "perimeter": [cell["perimeter"]],
                    }
                )],
                ignore_index=True,  
                )
        cellDFs.append(cellDF)

        df = pd.concat([df,
            pd.DataFrame(
                {
                    #csv format
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

    #write the analysis out to CSV
    try:
        df.to_csv(os.path.join(path, "summary.csv"), index=False)
        if not os.path.exists(os.path.join(path, "cells")):
            os.mkdir(os.path.join(path, "cells"))

        for i in range(len(cellDFs)):
            cellDFs[i].to_csv(os.path.join(path, "cells", files[i] + ".csv"), index=False)

    except PermissionError:
        print("Please close the summary.csv file. press any key to continue")
        input()
