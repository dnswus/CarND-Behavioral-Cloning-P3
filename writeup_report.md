# **Behavioral Cloning**

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image_center_lane_driving]: ./images/center_lane_driving.jpg "Center Lane Driving"
[image_non_flipped]: ./images/original_image.jpg "Original Image"
[image_flipped]: ./images/flipped_image.jpg "Flipped Image"
[image_left_camera]: ./images/left_camera.jpg "Left Camera"
[image_center_camera]: ./images/center_camera.jpg "Center Camera"
[image_right_camera]: ./images/right_camera.jpg "Right Camera"
[image_dirt_path]: ./images/dirt_path.jpg "Dirt Path"


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

### Architecture and Training Documentation

#### 1. Solution Design Approach

I started my model based on NVIDIA's model mentioned in the project description. At first, the models's mean square error of the training set is getting lower after each epoch, however, MSE of the validation set is not lowering. It means that the model was overfitting. To handle this, I added Dropout layers after each convolution and fully connected layers. It makes the MSE of the validation set getting lower after most epochs, although still not as low as MSE of the training set.

The next step was to run the simulator to see how well the car was driving around track one. I found that the car is not turning enough in corners and also not adjusting itself if it is very close to the side. It could be due to the lack of variations in the data set from my center lane driving. To improve this, I included the images from the side cameras with a correction of +/- 0.2. It improved the turning a lot. However, the car still had some difficulties in sharp turns and the turn at the first dirt path.

Then I added more training data at near the first dirt path and the next sharp turn. And also adjusted the correction to +/- 0.3 (tested many different values) and finally the car is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (lines 72-91) consisted of a convolution neural network with the following layers and layer sizes:

| Layer | Size | Output | Activation | Dropout |
| --- | --- | --- | --- | --- |
| Normalization | 160x320x3 (0 - 255) | 160x320x3 (-0.5 - 0.5) | None | None |
| Cropping | top 70, bottom 25 | 85x320x3 | None | None |
| Convolution | 24 filters, 5x5 kernel, 2x2 strides | 41x158x24 | RELU | 0.1 |
| Convolution | 36 filters, 5x5 kernel, 2x2 strides | 19x78x36 | RELU | 0.1 |
| Convolution | 48 filters, 5x5 kernel, 2x2 strides | 8x38x48 | RELU | 0.1 |
| Convolution | 64 filters, 3x3 kernel | 6x36x64 | RELU | 0.1 |
| Convolution | 64 filters, 3x3 kernel | 4x34x64 | RELU | 0.1 |
| Flatten | | 8,704 | None | None |
| Dense | | 100 | None | 0.2 |
| Dense | | 50 | None | 0.2 |
| Dense | | 10 | None | 0.2 |
| Dense | | 1 | None | None |

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![Center Lane Driving][image_center_lane_driving]

To augment the data sat, I also flipped images and angles thinking that this would solve the left turn bias with balanced data.

For example, here are the original image and flipped one:

![Original Image][image_non_flipped]
![Flipped Image][image_flipped]

I then used images from the side cameras (Left, Center, Right):

![Left Camera][image_left_camera]
![Center Camera][image_center_camera]
![Right Camera][image_right_camera]

After failing at sharp turns and the first dirt path, I collected more training data at the first dirt path:

![Dirt Path][image_dirt_path]

I finally randomly shuffled the data set and put 20% of the data into a validation set.

After the collection process, I had 11,302 data points. 80% for training (9,041 images from center camera). Add left and right cameras (27,123 images). I then preprocessed this data by flipping them (54,246 images). I used this training data for training the model. The validation set (2,260 images) helped determine if the model was over or under fitting. I used 5 epochs since more epochs did not improve the error rate. I also used generator to avoid running out of member with such a large dataset.
