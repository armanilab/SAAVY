#!/usr/bin/env python
# coding: utf-8
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.functional as TF
from torchScripts import utils
from torchScripts.engine import train_one_epoch, evaluate
import torch
import torch.utils.data
import json
import os
import PIL
import numpy as np
import random
import argparse



# Here, we are establishing the arguments that are required for user input when running the code
# There are four requirements:
parser = argparse.ArgumentParser()
parser.add_argument("--training", help="training folder")
parser.add_argument("--validation", help="validation folder")
parser.add_argument("--training_json", help="training json file")
parser.add_argument("--validation_json", help="validation json file")
 

args = parser.parse_args()


# setup logging so we can view the training progress in tensorboard
writer = SummaryWriter()



class OrganoidDataset(torch.utils.data.Dataset):
    """
    Used to train the model on the organoid dataset
    https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """

    def loadMasks(self, root):
        masks = {}
        fsImages = os.listdir(os.path.join(root))
        # load the json file containing the annotations
        with open(os.path.join(root, self.via)) as f:
           # parse it into json
            data = json.load(f)
            # Each json will contain a list of files, each file will conain a list of regions (annotated organoids) each region will have a list of x points and a list of y points
            # lots of looping to get the data into a format that we can use
            for key in data: # 
                # check if there is a OS file corresponding to the current file we have from json
                if data[key]["filename"] in fsImages:
                    if data[key]["regions"] == []:
                        continue
                    
                    # using the filename as a key, we set the value to the json list of regions
                    masks[data[key]["filename"]] = data[key]["regions"]

        return masks

# on construction, we load masks from file and the image paths form the json
    def __init__(self, root, via, shouldtransforms=False):
        self.root = root
        self.via = via
        self.shouldtransforms = shouldtransforms
        files = os.listdir(os.path.join(root))
        self.masks = self.loadMasks(root)
        self.imgs = list(self.masks.keys())

# for increased model generalization, we can augment data by flipping 
# this attempts to increase the number of training examples
    def transform(self, image, mask):


        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # Transform to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        return image, mask

# returns a mask and image for a given index used for training
    def __getitem__(self, idx):

        # get the image we are working with
        imagePath = os.path.join(self.root, self.imgs[idx])

        # create a new variable for the image that is converted to RGB
        img = PIL.Image.open(imagePath).convert("RGB")

        mask = self.masks[self.imgs[idx]]

        # mask is a dictionary of all x points and all y points. we have to convert these to a binary mask
        if self.shouldtransforms:
            img, target = self.transform(img, mask)
        masks = []

        # mask contains a list of regions (annotated organoids), we want to deal with each region separately
        for key in mask:
            points = key["shape_attributes"]
            x = points["all_points_x"]
            y = points["all_points_y"]
            # we can make a binary mask from this, but we need to know the size of the image
            # we can get this from the image itself
            width, height = img.size
            mask = PIL.Image.new("L", (width, height), 0)
            PIL.ImageDraw.Draw(mask).polygon(
                list(zip(x, y)), outline=1, fill=1
            )
            mask = np.array(mask, dtype=bool)
            masks.append(mask)
        numObjs = len(masks)
        boxes = []

        # creating a bounding box around the identified region/mask
        for i in range(numObjs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])


        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class - the organoid, so we can set the label to 1. For now, SAAVY cannot analyze more than 1 organoid type at a time. 
        # we can use the number of masks as the number of labels
        labels = torch.ones((numObjs,), dtype=torch.int64)

        # turn the masks (identified earlier) into an array of values
        masks = np.array(masks, dtype=np.uint8)
        # transform the masks to a tensor
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])

        # we need this because cocodataset has crowd (single instance) to be zero, we can just set this to zero and ignore it
        iscrowd = torch.zeros(
            (numObjs,), dtype=torch.int64
        )
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # creating a dictionary for the target with all information that we feed in/call from for the recognition
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["iscrowd"] = iscrowd
        target["area"] = area

        img = TF.to_tensor(img)

        return img, target

    # returns the length of the dataset
    def __len__(self):
        return len(self.imgs)

# main driver function that runs the training
def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

 
    def buildModel(numClasses):

        # start with pretrained MRCNN model form COOCO dataset
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

        # save the number of input features for the classifier
        inFeatures = model.roi_heads.box_predictor.cls_score.in_features

        # replace box predictor of pretrained mrcnn with new one that we will train
        model.roi_heads.box_predictor = FastRCNNPredictor(inFeatures, numClasses)
        inFeaturesMask = model.roi_heads.mask_predictor.conv5_mask.in_channels

        # recomend using 256 for hidden layer, can be changed if needed but between size of input and output
        hiddenLayer = 256

        # getting the masks for the prediction from the image! 
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            inFeaturesMask, hiddenLayer, numClasses
        )
        return model


    ### data augmentation, not needed for our dataset, however if there is difficulty with training
    # or generalization, this can be used to augment the data
    # def getTransform(train):
    #     transforms = []
    #     transforms.append(T.ToTensor())
    #     if train:
    #         #randomly flip the image and ground truth for data augmentation
    #         transforms.append(T.RandomHorizontalFlip(0.5))
    #     return T.Compose(transforms)

   

    dataset = OrganoidDataset(args.training, args.training_json, False)
    validationDataset = OrganoidDataset(args.validation, args.validation_json, False)

    # training/valdiaton dataloaders using our datasets
    dataLoader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=2, 
        shuffle=True, 
        num_workers=0, 
        collate_fn=utils.collate_fn 
    )

    validationDataLoader = torch.utils.data.DataLoader(
        validationDataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=utils.collate_fn
    )

    # 2 classes technically, background (which we don't consider) and organoid
    num_classes = 2

    # get get our custom MRCNN model and move it to GPU if we have one, otherwise CPU
    model = buildModel(num_classes)
    model.to(device)

    # select only those parameters of the model that need to be learned or adjusted 
    params = [p for p in model.parameters() if p.requires_grad]
    # standard optimizer with MRCNN default parameters. No scheduler.
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0001)

    # recomend starting with 15 epochs, increase if needed however model will overfit if you do too many epochs
    num_epochs = 15

    for epoch in range(num_epochs):

        train_one_epoch(
            model, optimizer, dataLoader, device, epoch, writer, print_freq=1
        )
        evaluate(model, validationDataLoader, writer, epoch, device=device)

    # exporting the final, trained model for the data
    # change the name of this as appropraite to your model 
    torch.save(model, "torchDebug01.1.pt")

#  only execute if ran as script, not if imported as module
if __name__ == "__main__":
    main()
    writer.flush()
