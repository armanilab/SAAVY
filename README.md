# Segmentation Algorithm to Assess the ViabilitY of 3D spheroid slices (aka SAAVY)

SAAVY was created for the purpose of predicting the viability percentage of 3D tissue cultures, cystic spheroids specifcally, according to brightfield micrscoppy plane slice images. SAAVY uses Mask R-CNN for instance segmentation to detect the spheroids in the focal plane within a provided image. Morphology is taken into account through our training with both live and dead spheroids. Live spheroids are distinctly spherical with noticeable edges whereas dead spheroids have a jagged outline from the apototic cell death. We based the viability algorithm on human expert assessmet and measure the intensity of the spheroid as compared to the background. Spheroids that have higher viabilities are typically closer in intensity values to that of the background, on average. Further, we include artificial noise in the backgrounds of the images to increase SAAVY's tolerance in the case of noisy biological backgrounds (i.e. matrix protein deposits, matrices loaded with foreign materials, and/or co-cultured cells creating a background).

SAAVY outputs the viability percent, average spheroid size, total count of spheorids included in the analysis, the total percent area of the image analyzed, and the average intensity value for the background. Our current code outputs the averages of each image, but maintains the ability to output specific viabilities, sizes, and intensity values for each invidivual spheroid identified in a given image.  


## Basic use
Follow these instructions for use of SAAVY with cycstic-type spheroids. Instructions for training images specifically according to your spheroid type (if of a differing morphology) are included below in the 'Fine Tune Model' section.

1. Clone this repository
```
git clone https://github.com/armanilab/SAAVY.git
```

All folders (inputs, outputs, training, etc.) must be in the SAAVY directory. The following code is written to call from the working directory.
```
cd SAAVY
```

Download the [model](https://drive.google.com/file/d/1NHOs9vxCup87TkMIZ8YFBuY9j8nx1NmH/view?usp=share_link) and move it into the SAAVY folder.

2. Create conda env 
```
conda create -n torch python="3.9"
conda activate torch
```

If you are running MAC, you will need to install pytorch with the following command
```
pip3 install torch torchvision torchaudio
```

Otherwise for WINDOWS
```
// GPU install requires CUDA toolkit https://developer.nvidia.com/cuda-toolkit
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
//CPU only, slower
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

Other requirements (MAC & WINDOWS)
```
pip3 install matplotlib scikit-learn pillow tqdm pandas opencv-python
```

After install, please check for the following:

* Python >= 3.9
* Pytorch >= 2.0
* Pillow >= 9.4.0
* matplotlib >= 3.7.1
* Conda installation
* (Optional but highly recomended) cuda-toolkit = 11.8 (ONLY IF RUNNING NVIDIA GPU)


4. Run the analysis 
```
python predict.py --input "YOUR FOLDER HERE" --output "CREATE A FOLDER HERE" --model "torchFinal.pt"
```

## Fine tune model
1. Download the [VIA image annotator 2.0.11](https://www.robots.ox.ac.uk/~vgg/software/via/)
   This will download a file to your computer with a name according to the version (via-2.0.11), open the folder and click via.html to run the program
   
2. Load images into dataset (add files in the annotator window) and create masks around them.
   Use either the circle, ellipse, or polygon tool to trace the edges of the spheroids of interest, whichever is appropriate to the shape of your sample.
   For example:

![.](https://images.duckarmada.com/5Qw1y2DW2t4s/direct.png)

3. Export as JSON. This will export the file to your default downloads folder/same as the Via Annotator Files.
![](https://images.duckarmada.com/Rmr7SCBEhTOX/direct.png)

4. **Name it** for example: `via_region_data.json`

5. Copy both JSON and annotated images into training directory, 

6. Repeat from step 2 for validation dataset

7. run `python training.py --training "training/" --validation "validation/"`

8. model will be saved to working directory


