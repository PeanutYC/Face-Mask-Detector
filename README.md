# Face-Mask-Detector
A face mask detector built with Tensorflow, OpenCV in Python


## Dataset
Dataset from [Data Flair](https://data-flair.training/blogs/download-face-mask-data/)

Consists of 1376 images with 690 images containing images of people wearing masks and 686 images with people without masks.

You are advised to increase the dataset images, especially those real images, instead of images with masks that are created artificially. Besides, you should increase the variation of masks colour for better accuracy.


## Face Detection
The files in the "face_detector" folder 
<ul>
  <li>a binary file containing trained weights</li>
  <li>a text file containing network configuration</li>
</ul>


## Face Mask Detection
Base Model - MobileNetV2 Model
(Fine-tuning) 


## Instructions to use the code
Download the dataset and add to your project. The folder is named as "dataset" with two subfolders "with_mask" and "without_mask"

### To train the model
Use the terminal console, type

`python train.py --dataset dataset`

I trained it with 10 epochs only, but you may further training it by modifying the variables in the code.

### To test the model
In the terminal console, type

`python test.py --image "YOUR IMAGE PATH"`

Please change the name according to your image path.


P/S This model can be further improved. 

References:
[SourceCode](https://github.com/sunnyahlawat1713/face-mask-detector-python)
