# Segmentation Algorithm to Assess the ViabilitY of 3D spheroid slices (aka SAAVY)

SAAVY was created for the purpose of predicting the viability percentage of 3D tissue cultures, cystic spheroids specifcally, according to brightfield micrscoppy plane slice images. SAAVY uses Mask R-CNN for instance segmentation to detect the spheroids in the focal plane within a provided image. Morphology is taken into account through our training with both live and dead spheroids. Live spheroids are distinctly spherical with noticeable edges whereas dead spheroids have a jagged outline from the apototic cell death. We based the viability algorithm on human expert assessmet and measure the intensity of the spheroid as compared to the background. Spheroids that have higher viabilities are typically closer in intensity values to that of the background, on average. Further, we include artificial noise in the backgrounds of the images to increase SAAVY's tolerance in the case of noisy biological backgrounds (i.e. matrix protein deposits, matrices loaded with foreign materials, and/or co-cultured cells creating a background).

SAAVY outputs the viability percent, average spheroid size, total count of spheorids included in the analysis, the total percent area of the image analyzed, and the average intensity value for the background. Our current code outputs the averages of each image, but maintains the ability to output specific viabilities, sizes, and intensity values for each invidivual spheroid identified in a given image.  




### Python requirements

* Python >= 3.9
* Pytorch >= 2.0
* Pillow >= 9.4.0
* matplotlib >= 3.7.1
* (Optional but highly recomended) cuda-toolkit = 11.8 
* Conda installation


## Basic use


Clone this repository

```
git clone https://github.com/armanilab/SAAVY.git
cd SAAVY
```
Download the [model](https://drive.google.com/file/d/1NHOs9vxCup87TkMIZ8YFBuY9j8nx1NmH/view?usp=share_link) and move it into the SAAVY folder.

Create conda env 


```
conda env create -n torch python="INPUT YOUR PYTHON VERSION"
conda activate torch
```
If you are running mac, you will need to install pytorch with the following command
```
pip3 install torch torchvision torchaudio
```
Otherwise for windows
```
// GPU install requires CUDA toolkit https://developer.nvidia.com/cuda-toolkit
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
//CPU only, slower
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

Other requirements
```
pip3 install matplotlib scikit-learn pillow tqdm pandas opencv-python
```


Run the analysis 
```
python predict.py --input "YOUR FOLDER HERE" --output "CREATE A FOLDER HERE" --model "torchFinal.pt"
```

## Fine tune model
1. download the [VIA image annotator 2.0.11](https://www.robots.ox.ac.uk/~vgg/software/via/)
2. load images into dataset and create polygon masks around them

![.](https://images.duckarmada.com/5Qw1y2DW2t4s/direct.png)

3. Export as JSON. **Name it**  `via_region_data.json`

![](https://images.duckarmada.com/Rmr7SCBEhTOX/direct.png)

4. Copy both JSON and annotated images into training directory, 

5. Repeat from step 2 for validation dataset

6. run `python training.py --training "training/" --validation "validation/"`

7. model will be saved to working directory


