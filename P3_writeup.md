# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images_output/train.png "Visualization"
[image2]: ./images_output/distribution.JPG "distribution"
[image3]: ./images_output/grayscale.png "Grayscaling"
[image4]: ./images_output/data_augmentation.png "data augmentation"
[image5]: ./images_output/tested_signs.png "Traffic Sign sample"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/HenryHXYao/CarND-P3/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 

One image in the train set is:

![alt text][image1]

The following image shows the distribution of classes in the training, validation and test set. The distributions among the three sets are nearly the same, however with some small difference.

![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because this approach can reduce the number of parameters in the first convolutional layer, thus saving memory and reducing training time. Furthermore, the experiments showed the grayscale images had better performance than the color image. 

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image3]

As a last step, I normalized the image data because normalization is a standard procedure for data preprocessing in CNN.

I decided to generate additional data because more data can help reduce overfitting and increase the overall prediction accuracy.

To add more data to the the data set, I referred to [Yann Lecun's paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) which is recommended above and used the following 3 image augmentation methods:

* Translation (randomly translate the image left/right and up/down in the range of [-2, +2] pixels)
* Rotation (randomly rotate the image in the range of [-10, +10] degrees)
* Scaling (randomly scale the image in the range of [0.9, 1.1] ratio)

Here is an example of an original image and an augmented image:

![alt text][image4]

8 fake images were generated for each image in the training set, therefore, the size of the augmented data set was 9 times of the original data set.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model is a modified version of 2-stage LeNet-5 which consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 gray normalized image   							| 
| Convolution 5x5     	| 80 filters, 1x1 stride, valid padding, outputs 28x28x80 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x80				|
| Convolution 5x5	    | 80 filters, 1x1 stride, valid padding, outputs 10x10x80				|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x80				|
|Flatten |      outputs 2000           |
| Fully connected | ouptuts	120|
| RELU					|												|
| Dropout					|												|
| Fully connected | ouptuts	84	|
| RELU					|												|
| Dropout					|												|
| Fully connected | ouptuts	43	|
| Softmax				| 			|

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an AdamOptimizer with batch size=128, epochs=30, learning rate = 0.001, keep probability of the dropout layer=0.5

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.989
* test set accuracy of 0.968

An iterative approach was chosen to improve the model, the following table shows the approach:

|Data Preparation|Network Architecture|Learning Rate|Epochs|Validation Accuracy 
| - | - | - | - | - |
|color normalized images|6 conv(5 * 5), relu, 2 * 2 pooling(stride = 2)<br>16 conv(5 * 5), relu, 2 * 2 pooling(stride = 2)<br>Dense(120), relu, dropout(keep = 0.7)<br>Dense(84), relu, dropout(keep = 0.7)<br>Dense(43)|0.001|30|0.953|
|gray normalized images|6 conv(5 * 5), relu, 2 * 2 pooling(stride = 2)<br>16 conv(5 * 5), relu, 2 * 2 pooling(stride = 2)<br>Dense(120), relu, dropout(keep = 0.7)<br>Dense(84), relu, dropout(keep = 0.7)<br>Dense(43)|0.001|30 | 0.953 |
|gray normalized images|20 conv(3 * 3), relu, 2 * 2 pooling(stride = 2)<br>16 conv(3 * 3), relu, 2 * 2 pooling(stride = 2)<br>Dense(120), relu, dropout(keep = 0.7)<br>Dense(84), relu, dropout(keep = 0.7)<br>Dense(43)|0.001|30 | 0.958 |
|color normalized images|20 conv(3 * 3), relu, 2 * 2 pooling(stride = 2)<br>16 conv(3 * 3), relu, 2 * 2 pooling(stride = 2)<br>Dense(120), relu, dropout(keep = 0.7)<br>Dense(84), relu, dropout(keep = 0.7)<br>Dense(43)|0.001|30 |0.953 |
|6 x augmented gray normalized images|20 conv(3 * 3), relu, 2 * 2 pooling(stride = 2)<br>16 conv(3 * 3), relu, 2 * 2 pooling(stride = 2)<br>Dense(120), relu, dropout(keep = 0.5)<br>Dense(84), relu, dropout(keep = 0.5)<br>Dense(43)|0.001|30 | 0.970 |
|6 x augmented gray normalized images|40 conv(3 * 3), relu, 2 * 2 pooling(stride = 2)<br>40 conv(3 * 3), relu, 2 * 2 pooling(stride = 2)<br>Dense(120), relu, dropout(keep = 0.5)<br>Dense(84), relu, dropout(keep = 0.5)<br>Dense(43)|0.001|30 | 0.980 |
|9 x augmented gray normalized images|80 conv(3 * 3), relu, 2 * 2 pooling(stride = 2)<br>80 conv(3 * 3), relu, 2 * 2 pooling(stride = 2)<br>Dense(120), relu, dropout(keep = 0.5)<br>Dense(84), relu, dropout(keep = 0.5)<br>Dense(43)|0.001|30 | 0.983 |
|9 x augmented gray normalized images|80 conv(3 * 3), relu, 2 * 2 pooling(stride = 2) <br>80 conv(3 * 3), relu, 2 * 2 pooling(stride = 2)<br>Dense(120), relu, dropout(keep = 0.5)<br>Dense(84), relu, dropout(keep = 0.5)<br>Dense(43)|0.0003| 50 | 0.981  |
|6 x augmented gray normalized images|80 conv(5 * 5), relu, 2 * 2 pooling(stride = 2)<br>80 conv(5 * 5), relu, 2 * 2 pooling(stride = 2)<br>Dense(120), relu, dropout(keep = 0.5)<br>Dense(84), relu, dropout(keep = 0.5)<br>Dense(43)|0.001|30 | 0.975 |
|9 x augmented gray normalized images|80 conv(5 * 5), relu, 2 * 2 pooling(stride = 2)<br>80 conv(5 * 5), relu, 2 * 2 pooling(stride = 2)<br>Dense(120), relu, dropout(keep = 0.5)<br>Dense(84), relu, dropout(keep = 0.5)<br>Dense(43)|0.001|30 | 0.989 |

Explanations are:
* As suggested by [Josh Tobin](http://josh-tobin.com/troubleshooting-deep-neural-networks.html), the LeNet with Convolution 5x5 and feature 6-16 (**the left and right numbers are the numbers of features at the first and second convolutional layers respectively**) was the first architecture that was tried. The original dataset was used without data augmentation and two preprocessing methods were tested. The normalization and gray normalization mthods achieved the same accuracy on the validation set to be 0.953. 
* Then I changed the convolution to be 3x3 and features to be 20-16. Gray normalization method achieved 0.958 accuracy and normalization method achieved 0.953 accuracy. Gray normalization method had better performance, therefore I used this preprocessing method in the following experiments.
* In the previous experiments, the accuracy on the training set was high but the accuracy on the validation set was low, which indicated over fitting. To solve the problem of overfitting, I reduced the keep probability of dropout layer to be 0.5 and added data augmentation. 6x augmented gray normalized images with 20-16 features achieved 0.970 accuracy while the model with 40-40 features achieved 0.980 accuracy. It showed increasing the feature numbers as well as adding data augmentation can increase the validation accuracy.
* Then I added more fake data into the training set. 9x augmented gray normalized images with 80-80 features achieved 0.983 accuracy. At this step, I tuned the learning rate to be 0.0003 and epochs to be 50. This change in the hyperparameters led to the validation accuracy to be 0.981.
* Finaly, I did experiments with Convolution 5x5 and features 80-80. 6x augmented gray normalized images achieved 0.975 accuracy and 9x 
augmented gray normalized images achieved 0.989 accuracy. The model with 0.989 validation accuracy was selected to be the final model.
 
### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 8 German traffic signs that I found on the web:

![alt text][image5]

The 5th image might be difficult to classify because there is a pole in front of the sign.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

|image No.| predicted label |true label|
|-|-|-|
|1|Double curve|Double curve |
|2|Speed limit (30km/h)|Speed limit (30km/h) |
|3|Ahead only|Ahead only |
|4|No entry|No entry |
|5|Pedestrians|Pedestrians |
|6|Priority road|Priority road |
|7|Go straight or right|Go straight or right |
|8|Stop| Stop|

The model predicts all the signs correctly, therefore the accuracy on these new images is 100%. This compares favorably to the accuracy on the test set of 0.968

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


