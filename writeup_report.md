# **Behavioral Cloning**

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

---

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on NVIDIA's network architecture documented at https://devblogs.nvidia.com/deep-learning-self-driving-cars/.

This model consists of 16 layers 5 convolution layers (line 74-82), 3 fully connected layers (line 85-89), and 8 dropout layers (1 after each of the convolution and fully connected layer).

The model uses RELU as activation function to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 72) and cropped out the top and bottom of the image to focus on the road (code line 73).

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers after each convolution and fully connected layer in order to reduce overfitting (lines 75-90).

The model was trained and validated on different data sets to ensure that the model was not overfitting (line 14). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 96).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, extra training data for the turn at the first dirt path.

I also included data from the left and right cameras to provide more training data. A correction of +/- 0.3 was used to adjust the steering values with left and right camera images.

Due of the large amount of training data, I used generator to work with the data and use `fit_generator` instead of `fit` from Keras to train the model.
