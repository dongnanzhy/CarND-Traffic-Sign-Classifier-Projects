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

[image1]: ./docs/data_explore_train.jpg "data_explore_train"
[image2]: ./docs/data_explore_dev.jpg "data_explore_dev"
[image3]: ./docs/data_explore_test.jpg "data_explore_test"
[image4]: ./docs/model_graph.png "model graph"
[image5]: ./docs/train_loss.png "train loss"
[image6]: ./docs/dev_accuracy.png "dev_accuracy"
[image7]: ./test_image/test_1.jpeg "Traffic Sign 1"
[image8]: ./test_image/test_2.jpeg "Traffic Sign 2"
[image9]: ./test_image/test_3.jpg "Traffic Sign 3"
[image10]: ./test_image/test_4.jpg "Traffic Sign 4"
[image11]: ./test_image/test_5.jpg "Traffic Sign 5"
[image12]: ./test_image/test_6.jpg "Traffic Sign 6"
[image13]: ./docs/visualize_cnn_1.png "visualize_cnn_1"
[image14]: ./docs/visualize_cnn_2.png "visualize_cnn_2"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/dongnanzhy/CarND-Traffic-Sign-Classifier-Projects/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the distribution of different classes across train/dev/test data.
![alt text][image1]
![alt text][image2]
![alt text][image3]
From above charts we can see the distribution of different classes is not a normal distribution, some traffic signs occur more frequently over others. We may also find that the distribution between train and test data are actually quite similar, while dev data distribution seems a little bit different. I think the reason may be number of images in dev data is not large, which may introduce higher variance on dev data.


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

During preprocessing, I only applied normalization by using `pixel = pixel / 255 - 0.5`. The reason I used normalization is for large neural network, the input features (which in our case is input image) should be with same mean/variance, to avoid converging to local minimum in training. Also, by applying normalization, model is less likely to overfit.

I did not apply grayscaling since I think changing first conv layer kernel size, the model can successfully handle color image, without losing information at preprocessing.

I did not perform other data augmentation techniques like adding random noise, Gaussian blur or cropping, while I think these are very useful techniques to enrich train data distribution and avoid overfitting. I will perform these techniques in further experiments.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 3x3     	| 5x5 kernel, 1x1 stride, same padding, outputs 32x32x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x16 				|
| Convolution 3x3     	| 3x3 kernel, 1x1 stride, same padding, outputs 16x16x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 8x8x32 				|
| Convolution 3x3     	| 3x3 kernel, 1x1 stride, same padding, outputs 8x8x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 4x4x64 				|
| Flattern	    | 1024x1.      									|
| Fully connected		| outputs 512x1  									|
| Dropout		| keep prob 0.5 									|
| Fully connected		| outputs 256x1  									|
| Dropout		| keep prob 0.5 									|
| Fully connected		| outputs 43x1  									|
| Softmax				|        									|

Tensorboard view of model graph is shown below.
![alt text][image4]

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used
* Adam optimizer
* batch size equals to 128
* number of epochs equals to 20
* learning rate equals to 0.001
* l2 loss ratio equals to 0.001
* dropout rate equals to 0.5.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set **loss** of 0.177
* validation set accuracy of 0.954
* test set accuracy of 0.949

Training loss and accuracy on dev data during training epochs can be seen below.
![alt text][image5]
![alt text][image6]

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
  I first choose LeNet architecture cause I used in the LeNet lab and it should be a reasonable baseline to try out.
* What were some problems with the initial architecture?
  When I trained with about 10 epochs, the model converges with accuracy on dev set at about 0.87, did not meet the requirements.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
  1. With initial LeNet architecture, I found the train loss seems to converge while dev set accuracy has not met requirements, So I guess the reason may be under fitting. The very straightforward thinking is to make model architecture more complex. So I double the number of kernels at first conv layer; change kernel size to 3*3 at second conv layer and double number of kernels at second conv layer; then add a third conv layer. Then I used 3 Fully Connected layers to get softmax vector.
  2. By making model architecture more complex, in case model suffering over fitting, I added dropouts for first two Fully Connected layers. The reason is by doubling number of kernels, the embedded feature map size is larger than before, so the Fully connected layers are more possible to overfit. Besides, I also added a l2 loss for Fully connected layer parameters to penalize if they get large.
* Which parameters were tuned? How were they adjusted and why?
  I also tuned all hyper parameters through experiments. Especially for number of epochs, according to training loss graph above, model converges at first 5-7 epochs, so it's not quite valuable to continue training, which means 15 epochs should be enough for this project.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
  My thoughts are included in the third bullet point. The reason convolution layer works as I think is it uses a conv operation to catch image information, while one kernel will share all parameters through the whole image. By doing this, each kernel will be trained to capture one characteristic of image, and with combination of all feature maps, we can embed image info and then make predictions.

If a well known architecture was chosen:
I did not try with a well known architecture but I think VGG or GoogleNet or ResNet may definitely get a quite low train loss, which leads more work on regularization to avoid overfitting.
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image7] ![alt text][image8] ![alt text][image9]
![alt text][image10] ![alt text][image11] ![alt text][image12]

The first 2 images might be difficult to classify because the size is rectangular, by resizing to 32*32, the traffic sign might be distorted.
The third image might be difficult because resizing may cause distortion, and there is a background mark in the middle of the traffic sign.
The forth and sixth image might be extremely difficult because in addition to reasons like third image, they are also blurred.
The fifth image should be easy to classify.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Road Work      		| Beware of ice/snow  									|
| Pedestrians     		| Dangerous curve to the right										|
| Speed limit (70km/h)		| Speed limit (20km/h)	            		|
| No passing	      		| Keep right					 				|
| No entry			| No entry      							|
| Children crossing			| Slippery Road      							|


The model was able to correctly guess 1 of the 6 traffic signs, which gives an accuracy of 17%. This result meets my guesses above and since I picked extremely noisy pictures on web, it's acceptable to have a lower accuracy than test image set.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 23rd cell of the Ipython notebook.

For the first image, the model is not sure whether this is a ice aware or road work, and the correct answer (road work) is ranked at 2nd place, with probability a little bit lower than the first. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .38         			| Beware of ice/snow   									|
| .33     				| Road Work										|
| .07					| Priority road											|
| .07	      			| Dangerous curve to the right					 				|
| .05				    | No entry    							|

For the second image, I actually not quite sure whether the true label is Pedestrian or Children Crossing. Anyway, the model is relatively confident at predicting as Dangerous Curve, which is not correct. The second rank prediction with a probability 0.25 seems reasonable.

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .72         			| Dangerous curve to the right   									|
| .25     				| Children crossing										|
| .017					| Beware of ice/snow											|
| .002	      			| Slippery road					 				|
| .0017				    | Road narrows on the right   							|

For the third image, the model is relatively confident at predicting as 20 km/h speed limit, which is not correct. The second rank prediction with a probability 0.2 is correct.

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .79         			| Speed limit (20km/h)  									|
| .20     				| Speed limit (70km/h)										|
| .006					| Speed limit (30km/h)											|
| .003	      			| Speed limit (120km/h)					 				|
| .00005				    | No entry    							|

For the forth image, I actually not quite sure what the true label is. But the model does not have a high confident on predicting, and all the top 5 predictions seem to be not correct.

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .5         			| Keep right  									|
| .15     				| End of no passing										|
| .12					| Priority road										|
| .05	      			| Roundabout mandatory					 				|
| .036				    | Go straight or right    							|

For the fifth image, the image is quite clear and the traffic sign is located at center of the image, so it is easy to classify. The model has quite high confidence on predicting as No Entry, which is correct.

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .99         			| No entry 									|
| .0     				| Stop										|
| .0					| No passing										|
| .0	      			| Vehicles over 3.5 metric tons prohibited					 				|
| .0				    | Dangerous curve to the left    							|

For the sixth image, I actually not quite sure what the true label is. The model has a relatively high confidence on prediction as Slippery road, which I don't think is correct. And all the top 5 predictions seem to be not correct.

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .77         			| Slippery road									|
| .08     				| Go straight or left									|
| .06					| End of all speed and passing limits											|
| .02	      			| Roundabout mandatory					 				|
| .02				    | Speed limit (50km/h)   							|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
I random choose one image from train data, which is a speed limit of 50 km/h.
The first conv layer feature maps are as below.
![alt text][image13]
The second conv layer feature maps are as below.
![alt text][image14]

1. I found the second conv layer feature maps are hard to identify. One reason may be the size of feature maps at conv 2 layer is 8*8, which is quite small. Also, the higher level feature maps seem to be hard to visualize, while we can use other way like maximally activating patches to visualize.
2. For the first conv layer feature maps. For examples, feature map 4, it can effectively capture bounding circle of the traffic sign and highlight it in activation; feature map 14, it can identify the number *0* and *5* with outlines.
