# if you're noting information overviewing the following model, please highlight that here
#!/usr/bin/env python
# coding: utf-8


# what is this debug cell for? we're publishing final code, we shouldn't have any of this - KT
# In[1]:

# make sure that you're loading ALL packages that you use throughout the document here first so that people who look at it to use can easily check for installation - KT
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import torch
import torch.utils.data
import json
import os
# why use PIL? Isn't this depricated after Python 2? We load pillow, so shouldn't we just run from pillow? - KT
import PIL
import numpy as np
import random
import argparse

'''
The code isn't annotated... 
I'm going to annotate based on my understanding of what you are doing, if I am wrong please correct.
'''

# Here, we are establishing the arguments that are required for user input when running the code
# There are four requirements:
parser = argparse.ArgumentParser()
parser.add_argument("--training", help="training folder")
parser.add_argument("--validation", help="validation folder")
parser.add_argument("--training_json", help="training json file")
parser.add_argument("--validation_json", help="validation json file")
# Are the training and validation json files the same? - KT 

args = parser.parse_args()
# if the following is retired code, please remove it - KT
# import scripts.pytorchVisionScripts.utils as utils
# from scripts.pytorchVisionScripts.engine import *

from torch.utils.tensorboard import SummaryWriter

# assuming that you're inputting this for a specific read/write function to wrap the ML? - KT
writer = SummaryWriter()

# keep all imports together at the beginning of the document - KT
# and note the grouping of imports and what they are for - KT
import torchvision.transforms.functional as TF
from torchScripts import utils
from torchScripts.engine import train_one_epoch, evaluate

# In[2]:

# note what this is for and what you are doing here - KT
class OrganoidDataset(torch.utils.data.Dataset):
    """
    flow courtesy of pytorch.org's finetuning documentation
    # if there is a link, I would inlcude the link here to easily get to the documentation that you are talking about - KT
    """

    def loadMasks(self, root):
        masks = {}
        fsImages = os.listdir(os.path.join(root))
        # what are you doing here? This is what is erroring when I try running the training script, so you might need to change this for a more efficient manner of calling the masks
        with open(os.path.join(root, self.via)) as f:
            # loading the data from the json file upload
            data = json.load(f)
            for key in data: # iterating through every dictionary(?) in the uplaoded json file?? Is this true? What is the "key" here bc the json file is tricky to understand what you're looping through
                if data[key]["filename"] in fsImages:
                    # check if regions exist is empty and if so remove the image from the list
                    # TODO: make it so that null images can be used for training
                    # is that TODO still outstanding? If it's not, clear it from the code - KT
                    if data[key]["regions"] == []:
                        # self.imgs.remove(data[key]["filename"])
                        # remove outdated code - KT
                        pass
                    else:
                        masks[data[key]["filename"]] = data[key]["regions"]
# Why is this chunk here? If it's elftover from your code build where you are debugging, can you remove this? We want to publishe/release cleaned code
        return masks

# note what this function is doing - KT 
    def __init__(self, root, via, shouldtransforms=False):
        self.root = root
        self.via = via
        self.shouldtransforms = shouldtransforms
        # load all image files, sorting them to
        # ensure that they are aligned
        files = os.listdir(os.path.join(root))
        # ignrore all .json files
        # note why - KT
        self.masks = self.loadMasks(root)
        self.imgs = list(self.masks.keys())

# note what this function is doing - KT
# remove all of the retired code - KT
# why are you resizing all of these things? - KT
    def transform(self, image, mask):
        # # Resize
        # resize = transforms.Resize(size=(520, 520))
        # image = resize(image)
        # mask = resize(mask)

        # # Random crop
        # i, j, h, w = transforms.RandomCrop.get_params(
        #     image, output_size=(512, 512))
        # image = TF.crop(image, i, j, h, w)
        # mask = TF.crop(mask, i, j, h, w)

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

# what does this section do? - KT
    def __getitem__(self, idx):

        # get image path?
        imagePath = os.path.join(self.root, self.imgs[idx])

        # create a new variable for the image that is converted to (or from) RGB ? - KT
        img = PIL.Image.open(imagePath).convert("RGB")

        # why is this mask differnt than the other masks above? would that interfere with calling the other mask varioable (even though these are within functions)?
        # if they are different, I would personally rename the variables so that others who read through the code don't get confused
        mask = self.masks[self.imgs[idx]]

        # mask is a dictionary of all x points and all y points. we have to convert these to a binary mask
        if self.shouldtransforms:
            img, target = self.transform(img, mask)
        masks = []

        # this loop is.... fill this out - KT
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

        # creating a bounding box around the identified region/mask - KT
        for i in range(numObjs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])


        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class, why? --> Is this a note to yourself? - KT
        # we can use the number of masks as the number of labels
        labels = torch.ones((numObjs,), dtype=torch.int64)

        # turn the masks (identified earlier) into an array of values
        masks = np.array(masks, dtype=np.uint8)
        # transform the masks to a tensor
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        # labelling each box with an ID number that indexes with the number of each mask - KT
        image_id = torch.tensor([idx])

        # we need this because cocodataset has crowd (single instance) to be zero
        # iscrowd is used for ... - KT
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

        # converting the image to tensor (again??) - KT
        # does this take into account other data/bounding boxes that you just created earlier?
        img = TF.to_tensor(img)

        return img, target

    # what is this funciton for? - KT
    def __len__(self):
        return len(self.imgs)

# this function does ... - KT
def main():
    # do we need different commands for GPU v CPU in the instructions if there is this check for the devices here already - KT
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # creating the model taking in a variable containing the number of classes - KT
    def buildModel(numClasses):

        # calling the prebuilt model architecture from Mask R-CNN - KT
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

        # what's going on here? - KT
        inFeatures = model.roi_heads.box_predictor.cls_score.in_features

        # we are also calling the bounding boxes from Fast R-CNN (is this different?) to give us the outlines of the identified spheroids - KT
        model.roi_heads.box_predictor = FastRCNNPredictor(inFeatures, numClasses)

        # obtaining all of the features in the mask - KT
        inFeaturesMask = model.roi_heads.mask_predictor.conv5_mask.in_channels

        # setting the total number of hidden layers for the NN architecture
        # how did you decide on this value? - KT
        hiddenLayer = 256

        # getting the masks for the prediction from the image! 
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            inFeaturesMask, hiddenLayer, numClasses
        )
        return model


    ### if this section is commented out, can we delete it? - KT
    ### if eventually this could be useful and you want to keep it, please note the instance in which this should be uncommented - KT
    # def getTransform(train):
    #     transforms = []
    #     transforms.append(T.ToTensor())
    #     if train:
    #         #randomly flip the image and ground truth for data augmentation
    #         transforms.append(T.RandomHorizontalFlip(0.5))
    #     return T.Compose(transforms)

    # split the dataset in train and test set
    def trainTestSplit():
        folder = "trainingData"
        # check if trainingData/train and trainingData/test exist
        if not os.path.exists(os.path.join(folder, "train")):
            os.mkdir(os.path.join(folder, "train"))

        if not os.path.exists(os.path.join(folder, "test")):
            os.mkdir(os.path.join(folder, "test"))

        # copy 10% of the images to the test folder
        for file in os.listdir(os.path.join(folder, "images")):
            if np.random.rand(1) < 0.1:
                os.rename(
                    os.path.join(folder, "images", file),
                    os.path.join(folder, "test", file),
                )
                os.rename(
                    os.path.join(folder, "via_region_data.json"),
                    os.path.join(folder, "test", "via_region_data.json"),
                )
            else:
                os.rename(
                    os.path.join(folder, "images", file),
                    os.path.join(folder, "train", file),
                )
                os.rename(
                    os.path.join(folder, "via_region_data.json"),
                    os.path.join(folder, "train", "via_region_data.json"),
                )

    # we have a train test split so we dont need to do this
    ### if we don't need to do the above, why do we have this in the code? I would remove this and/or comment it out with a note that a user can uncoment this if they decide to edit things here


    dataset = OrganoidDataset(args.training, args.training_json, False) # i would specify training dataset in the name since you refer to dataset earlier in the OrganoidDataset code, explicit is better for others to follow
    validationDataset = OrganoidDataset(args.validation, args.validation_json, False)

    # loading in the data from ???
    dataLoader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=2, 
        shuffle=True, 
        num_workers=0, 
        collate_fn=utils.collate_fn # do you need a comma here ?? There's one at the end of the other code... maybe this is the training data error?????? - KT
    )

    # stay consistent in how you format your code, this is different than the above 
    # I changed the above to match since I usually code in this manner for clarity - KT
    validationDataLoader = torch.utils.data.DataLoader(
        validationDataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=utils.collate_fn,
    )

    # I thought you said there was one class earlier? - KT
    num_classes = 2

    # this step packages the model for export? - KT
    model = buildModel(num_classes)
    model.to(device)

    # what is this and why are you introducting this after you already create and export the model?? - KT
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0001)

    # explain how you got to this choice of epochs - KT
    # is this something that other would have to change to train their own data? - KT
    # if yes to the above, make a note of this and what epoch number you recommend starting off with - KT
    num_epochs = 15

    for epoch in range(num_epochs):

        train_one_epoch(
            model, optimizer, dataLoader, device, epoch, writer, print_freq=1
        )
        # lr_scheduler.step() -- if you don't use this, remove this from the code - KT
        evaluate(model, validationDataLoader, writer, epoch, device=device)

    # exporting the final, trained model for the data
    # change the name of this as appropraite to your model 
    torch.save(model, "torchDebug01.1.pt")

# what does this section do? please explain - KT
if __name__ == "__main__":
    main()
    writer.flush()
