# **Behavioral Cloning** 

[//]: # (Image References)

[centre_image]: ./examples/centre_image.jpg "Sample Centre Camera Image"
[flipped_centre_image]: ./examples/flipped_centre_image.jpg "Sample Flipped Centre Camera Image"
[left_image]: ./examples/left_image.jpg "Sample Left Camera Image"
[right_image]: ./examples/right_image.jpg "Sample Right Camera Image"
[loss_data]: ./examples/loss_data.png "Training and Validation Loss Data"

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report
---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results
* output/run1.mp4 video captured while running the car in autonomous mode using model.h5
* Training and Validation loss data in examples/data_loss.png
* Sample camera images and flipped image for centre camera image

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 

python drive.py model.h5

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a five convolution neural network layers with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 84-100) 

The model includes RELU layers to introduce nonlinearity (code line 84), and the data is normalized in the model using a Keras lambda layer (code line 82). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layer after each convolutional layer and some of the Dense layers with dropout rate of 0.1 to reduce overfitting. Also the model was trained and validated on different data sets. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

For training the model, multiple training samples were used for same track area. e.g. at steep curved track, data was obtained for various scenarios.

After 7 epochs, training loss was 0.0379 and validation loss was 0.0330.
Below figure represents mapping of training and validation loss with number of epochs:

![alt text][loss_data] 

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 113).

#### 4. Appropriate training data

Training data consists of camera images (left, right and centre camera images) and steering angle. 

Below are the sample camera images I used (including flipped centre camera image):

Sample centre camera image:
![alt text][centre_image]

Sample flipped centre camera image:
![alt text][flipped_centre_image]

Sample left camera image:
![alt text][left_image]

Sample right camera image:
![alt text][right_image] 

While collecting training data, I used a combination of center lane driving, recovering from the left and right sides of the road by running through same portion of the track multiple times with different steering angles.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use a combination of convolutional neural network and connected networks and provide sufficient input data to the model to get correct model data.

My first step was to use a convolution neural network model similar to the traffic sign classifier project which is used for classification of input images. I thought this model might be appropriate because in behaviour cloning project also, input data is set of images and labels are the steering angles.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. Then for each input image, I used flipped image data to provide more input for improving accuracy of the model.

Also, left and right camera images were also used for training. Steering angle was calculated using correction factor of 0.2 as:

steering angle for left image = steering angle + 0.2
steering angle for right image = steering angle - 0.2

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track in the beginning. For those spots, I had to collect data multiple times with different steering angles. e.g. to prevent vehicle from going outside of the road on the left side, turning the vehicle right side quickly, etc. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded single lap on track one using center lane driving.

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to come back to the centre of the lane.

To augment the data sat, I also flipped images and angles thinking that this would improve the accuracy of the model. I also used left and right camera images for training.

There were 559,419 number of total trainable parameters for the model as shown below:

Layer (type)                 Output Shape              Param #
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 70, 320, 3)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 33, 158, 24)       1824
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 15, 77, 36)        21636
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 6, 37, 48)         43248
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 4, 35, 64)         27712
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 2, 33, 64)         36928
_________________________________________________________________
flatten_1 (Flatten)          (None, 4224)              0
_________________________________________________________________
dense_1 (Dense)              (None, 100)               422500
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11
=================================================================
Total params: 559,419
Trainable params: 559,419
Non-trainable params: 0

I finally randomly shuffled the data set and put 20% of the data into a validation set. 
The ideal number of epochs was 5.
I used an adam optimizer so that manually training the learning rate wasn't necessary.

output directory contains the video captured by running the vehicle in autonomous mode using the data (model.h5) captured after training the model.