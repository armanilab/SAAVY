# Segmentation Algorithm to Assess the ViabilitY of 3D spheroid slices (aka SAAVY)

SAAVY was created for the purpose of predicting the viability percentage of 3D tissue cultures, cystic spheroids specifically, according to brightfield microscopy plane slice images. SAAVY uses Mask R-CNN for instance, segmentation to detect the spheroids in the focal plane within a provided image. Morphology is taken into account through our training with both live and dead spheroids. Live spheroids are distinctly spherical with noticeable edges, whereas dead spheroids have a jagged outline from the apototic cell death. We based the viability algorithm on human expert assessment and measured the intensity of the spheroid as compared to the background. Spheroids that have higher viabilities are typically closer in intensity values to that of the background, on average. Further, we include artificial noise in the backgrounds of the images to increase SAAVY's tolerance in the case of noisy biological backgrounds (i.e. matrix protein deposits, matrices loaded with foreign materials, and/or co-cultured cells creating a background).

SAAVY outputs the viability percent, average spheroid size, total count of spheroids included in the analysis, the total percent area of the image analyzed, and the average intensity value for the background. Our current code outputs the averages of each image, but maintains the ability to output specific viabilities, sizes, and intensity values for each individual spheroid identified in a given image.  

The following document includes instructions for using SAAVY using example data we provide (based on our [manuscript, arXiv](https://arxiv.org/abs/2311.09354)) and uploading your own data for training and analysis. Our example data is cycstic spheroids with clear and noisy backgrounds. The full image dataset is hosted on [Zenodo](https://doi.org/10.5281/zenodo.10086368). Instructions for training images specifically according to your spheroid type (if of a differing morphology) are included below in the 'Fine Tune Model' section.


## Instructions for Use
Note: All proceeding steps require Conda installation. 

Check for conda installation **OR** follow [directions to install conda](https://conda.io/docs/user-guide/install/) 

1. Clone this repository using your device terminal or IDE of choice:
```
git clone https://github.com/armanilab/SAAVY.git
```

2. Enter the SAAVY directory in terminal: 
```
cd SAAVY
```
All folders (inputs, outputs, training, etc.) must be in the SAAVY directory. The following code is written to call from the working directory.

3a. If you are following our example/using similar cycstic spheroids, download the [model](https://drive.google.com/file/d/1NHOs9vxCup87TkMIZ8YFBuY9j8nx1NmH/view?usp=share_link) and save it to the SAAVY folder.

3b. If you are training your own images, skip this step.

4. Create virtual environment:
```
conda create -n torch python="3.9"
conda activate torch
```

5. Install packages:
If you are running **MAC**, you will need to install pytorch with the following command:
```
pip3 install torch torchvision torchaudio
```

Otherwise for **WINDOWS**:
```
// GPU install requires CUDA toolkit https://developer.nvidia.com/cuda-toolkit
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
//CPU only, slower
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

Other requirements (**MAC & WINDOWS**)
```
pip3 install matplotlib scikit-learn pillow tqdm pandas opencv-python
```

After install, please check for the following:

* Python = 3.9
* Pytorch >= 2.0
* Pillow >= 9.4.0
* matplotlib >= 3.7.1
* (Optional but highly recomended) cuda-toolkit = 11.8 (ONLY IF RUNNING NVIDIA GPU)

6a. If using our example images and training data, or if **your samples are of cystic spheroid type**, run SAAVY viability analysis using:
```
python predict.py --input "YOUR FOLDER HERE" --output "CREATE A FOLDER HERE" --model "torchFinal.pt"
```
And stop here - you're done!


## Fine tune the model
If using your own images, **of a cryptic spheroid type**, follow the following steps --> 

6b. Download the [VIA image annotator 2.0.11](https://www.robots.ox.ac.uk/~vgg/software/via/)

   This will download a file to your computer with a name according to the version you downloaded (via-2.0.11)
   
7. Open the VIA folder and open the ```via.html``` file to run the program. It will show in a new browser window.
   
8. Load images into VIA (add files button in the annotator window).

   Images must be PNG of JPG format. We suggest opening images on your device and export from the viewer to PNG or JPG format. 
   We used 30 images for our balanced training/validation image subset with an 80%/20% split.

9. Create masks around the regions (spheroids/organoids) you are interested in having SAAVY analyze. Use the polygon tool to trace the edges of the spheroids of interest. 

   For example: ![.](https://images.duckarmada.com/5Qw1y2DW2t4s/direct.png)

10. Export as JSON. This will export the file to your default downloads folder/same as the Via Annotator Files. Go to the annotations menu and use the JSON dropdown option.
    ![](https://images.duckarmada.com/Rmr7SCBEhTOX/direct.png)

11. Rename the annotator JSON file according to which file type (training or validation)
    
12. **You will have to do this twice**: once for your training data, and once for your validation data. Repeat steps 8-11 for validation data. You should have two folders after, training and validation, each with their own images and annotation json files.

13. Move the annotator JSON file and training images into a training directory i.e. "trainingData" Folder

14. Move the validation images into a validation directory i.e. "validationData" Folder

15. Install packages for training script:
    ```
    pip3 install pycocotools tensorboard
    ```
    
16. Run
    ```
    python training.py --training "TRAINING FOLDER" --validation "VALIDATION FOLDER" --training_json "TRAINING ANNOTATIONS JSON" --validation_json "VALIDATION ANNOTATIONS JSON"
    ```

17. The model will be saved to your working directory to be used with the instructions 1-5 above

