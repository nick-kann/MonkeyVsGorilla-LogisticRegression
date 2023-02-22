
# Monkey vs Beef Carpaccio Classification Model
This is a binary classification model using logistic regression that predicts whether a given 64 x 64 pixel image is of a monkey or beef carpaccio.

To classify an image using this model, the image is flattened into a column vector and standardized by dividing each value by 255. The model treats each RGB value of the image as a node in a single layer and passes its weight into the sigmoid activation function to calculate the predicted y value (yhat). If yhat is above 0.5, the model predicts the image to be a monkey. If yhat is below 0.5, the model predicts the image to be beef carpaccio. \
<br/><br/>
Here's a diagram that might help visualize the process: 

![Logistic Regression Model](https://i.imgur.com/sIj1U8d.jpeg) 
<br/><br/>

## Usage
It is recommended to run the model in a virtual environment avoid conflicts with other projects that may be using different versions of the same libraries.<br/><br/>
The learning rate and number of iterations can be adjusted in the model's parameters to fine-tune the model's performance and achieve better test accuracy results.<br/><br/>
The model can be used for custom images by adding an image to the same directory and changing the "test_image" string to match the filename of the custom image.

## Example
###### Train accuracy: 94.90% <br/>Test accuracy: 90.78% <br/> (rounded to the nearest hundredth)
<img src="https://i.imgur.com/JEW0KBM.png" alt="beef carpaccio prediction" width="400"/><img src="https://i.imgur.com/qwbskxH.png" alt="monkey prediction" width="400"/><img src="https://i.imgur.com/F93uUBE.png" alt="cost function curve" width="400"/>



## Install dependencies
```
pip install numpy
pip install pillow
pip install matplotlib
```

## Author
Nicholas Kann / [@Butter-My-Toast](https://github.com/Butter-My-Toast "Butter-My-Toast's github page")


## Credits
#### This project uses the following datasets:
- [Monkey dataset](https://images.cv/dataset/monkey-image-classification-dataset) (images.cv)
- [Beef carpaccio dataset](https://images.cv/dataset/beef-carpaccio-image-classification-dataset) (images.cv)
#### This project uses the following libraries:
- [numpy](https://github.com/numpy/numpy) (BSD-3-Clause License) - [License](https://github.com/numpy/numpy/blob/main/LICENSE.txt)
- [Pillow](https://github.com/python-pillow/Pillow) (HPND License) - [License](https://github.com/python-pillow/Pillow/blob/main/LICENSE)
- [matplotlib](https://github.com/matplotlib/matplotlib) (MDT License) - [License](https://github.com/matplotlib/matplotlib/blob/main/LICENSE/LICENSE)
