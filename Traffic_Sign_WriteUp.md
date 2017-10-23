#**Traffic Sign Recognition**

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[class-histogram]:   images/class-histogram.png   "Class Histogram"
[data-augmentation]: images/data-augmentation.png "Data Augmentation"
[dataset-histogram]: images/dataset-histogram.png "Dataset Histogram"
[preprocessing]:     images/preprocessing.png     "Preprocessing methods"

[lenet]: http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf
[shang]: https://arxiv.org/abs/1603.05201

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my
[project code](https://github.com/corrado9999/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

The following chart shows how data are distributed among the three datasets. Circa 65% of data is
being used for training, 10% for validation and 25% for testing.

![Dataset histogram][dataset-histogram]

The second chart shows how each class is represented in the three datasets (bars are normalized
per-dataset in order to make them comparable). With this visualization is clear that the
distribution of the classes is far from uniform, being some classes much less represented than
others. Notably, "keep right" is 5 times more present than "keep left". However, the distribution is
more or less the same in the three datasets, allowing this way to test the model in the same
conditions it was trained.

![Class histogram][class-histogram]\


###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.  (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

According to [Sermanet&LeCunn][lenet], grayscale images performs better than color images (probably
because, in trafic signs, colors are used more to attract attention than to really distinguish them).
So I concentrated on such images.

In order to convert them to grayscale, again following the cited paper, I used the YUV colorspace
keeping only the luminance component (Y).

Unfortunately, the lightning conditions of the dataset are very different and often over-saturated
or under-saturated (see [!cell8]). Therefore I decided to apply an additionaly step to improve the
signs readability. I tried the following methods:

 1. Gray: Y only
 2. Enhanced gray: (Y/255)^0.375 (to enhance differences in dark areas)
 3. Enhanced HSV: Y of the image obtained by raising the saturation and value to 0.5.
 4. Equalized gray: adaptive histogram equalization (from scikit-image package) applied to Y.

After each method, a normalization step to obtain a zero-mean 1/3-stddev is performed followed by
a clipping between -1 and 1 (because following scikit-image functions require float images in that
interval.

A comparison between the original images and the 4 methods is shown in the following images:

![Preprocessing method][preprocessing]\


Many original images are barely if not completely incomprehensible (2-2, 2-3, 2-10, 6-4, 4-10,
10-9). Method 1 already shows great improvements, but the contrast is still low in 6-4 and 10-9.
Methods 2 and 3 slightly improve one of them each, but they are not completely satisfactorial yet.
Method 4, although slightly altering the background textures, makes quite acceptable all the
critical cases. For this reason, I chose it as definitive preprocessing step.

To make the learning more robust and able to generalize, I created a class to dynamically increase
the dataset size by applying a random affine transform and adding a random noise. The parameter
chosen for this steps have been:

 - Zoom factor between 0.8 and 1.2 (bivariate gaussian, x and y independetely, with one-mean and
   0.2/3 stddev)
 - Rotation between -30 and 30 degrees (gaussian with 0-mean and 10-stddev)
 - Shearing between -30 and 30 degrees (gaussian with 0-mean and 10-stddev)
 - Additive gaussian noise with 0-mean and 0.1-stddev.

Some randomly chosen examples of images before and after data augmentation are shown below.

![Data augmentation][data-augmentation]

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.). Consider including a diagram and/or table describing the final model.

My final model consists in a sequence of convolutional layers followed by a MLP.
Inspired by [Sermanet&LeCunn][lenet]:
 1. Convolutions are 5x5
 2. Convolutions are always followed by max-pooling
 3. All the convolutional layers are fed to the MLP, not only the last one.

Following the suggestion of [Shang et al.][shang] I used a Concatenated ReLU in the first
convolutional layer. This makes the first layer training faster and more robust.

The following table summarizes the model:

| ID              | Layer                 | Output shape  | Linked to      | #params |
|:----------------|:----------------------|:--------------|:---------------|--------:|
| layer1/input    | Input                 | 32 x 32 x   1 | layer1/conv    |       0 |
| layer1/conv     | Convolution 5x5       | 32 x 32 x  16 | layer1/crelu   |     400 |
| layer1/crelu    | CReLU                 | 32 x 32 x  32 | layer1/pool    |       0 |
| layer1/pool     | Max Pooling 2x2       | 16 x 16 x  32 | layer1/pool2   |       0 |
|                 |                       |               | layer2/conv    |         |
| layer1/pool2    | Max Pooling 4x4       |  4 x  4 x  32 | flat           |       0 |
| layer2/conv     | Convolution 5x5       | 16 x 16 x  64 | layer2/relu    |   51200 |
| layer2/relu     | ReLU                  | 16 x 16 x  64 | layer2/pool    |       0 |
| layer2/pool     | Max Pooling 2x2       |  8 x  8 x  64 | layer2/pool2   |       0 |
|                 |                       |               | flat           |         |
| layer2/pool2    | Max Pooling 2x2       |  4 x  4 x  64 | layer2/conv    |       0 |
| layer3/conv     | Convolution 5x5       |  8 x  8 x 128 | layer3/relu    |  204800 |
| layer3/relu     | ReLU                  |  8 x  8 x 128 | layer3/pool    |       0 |
| layer3/pool     | Max Pooling 2x2       |  4 x  4 x 128 | flat           |       0 |
| flat            | Concat&Flat           | 3584          | layer4/dense   |       0 |
| layer4/dense    | Fully connected       | 1024          | layer4/relu    | 3671040 |
| layer4/relu     | ReLU                  | 1024          | layer4/dropout |       0 |
| layer4/droupout | Dropout               | 1024          | layer5/dense   |       0 |
| layer5/dense    | Fully connected       | 256           | layer5/relu    |  262400 |
| layer5/relu     | ReLU                  | 256           | layer5/dropout |       0 |
| layer5/droupout | Dropout               | 256           | layer6/dense   |       0 |
| layer6/dense    | Fully connected       | 64            | layer6/relu    |   16448 |
| layer6/relu     | ReLU                  | 64            | layer6/dropout |       0 |
| layer6/droupout | Dropout               | 64            | logits/dense   |       0 |
| logits/dense    | Fully connected       | 43            | logits/softmax |    2795 |
| logits/softmax  | Softmax normalization | 43            |                |       0 |

The total number of parameters is 4,209,083.

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

For the training I adopted AdamOptimizer and I implemented a very simple learning rate modulation
(by lowering the rate when reaching certain validation accuracies), dropout modulation and early
stopping. I also saved the best solution found at each epoch in order to load it afterwards.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps.  Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

* training set accuracy of 100.0%
* validation set accuracy of 99.0%
* test set accuracy of 97.3%

I began with the LeNet network from the LeNetLab, without data augmentation and simply applied to
grayscale images. The validation accuracy was no higher than 85%. I then tried to add some dropout
and the accuracy dramatically reduced!!! I interpreted this as a sign of too few parameters, because
the aim of dropout is to add redundancy, but clearly there was not enough space for it.

I therefore tried to add more channels to the convolutional layers, following the indications of
[Sermanet&LeCunn][lenet], but the time needed to perform the training was too long, and I did not
like the idea of addressing the problem by brute force, not at the very beginning.  So I decided to
"go deeper" instead of "wider". Hence I put an additional convolutional layer and rearranged the
channel depth. This of course introduced more parameters to be learnt, but I was able to stay below
1 minute per epoch on my two-cores i7-4500U@1.80GHz. This permitted to me to reach 90% validation
accuracy within 20 epochs, but it was not enough, and also dropout was still counterproductive.

Adding fast-farward connections as in [Sermanet&LeCunn][lenet] (and rearranging the depth of the MLP
in order to take into account properly the new inputs) was what the network needed to reach the
required accuracy (although barely) and to make the dropout effective.

Introducing the Concatenated ReLU in the first convolutional layer and implementing the data
augmentation class gave the sprint to overcome the 95% even before the tenth epoch.

By increasing again the depth of the network and after introducing the decreasing learning rate rule
and early-stopping I was able to succesfully train for many more epochs (up to 200) and to reach 99%
of validation accuracy. I also noticed that, by progressively increasing the keep probability of the
dropout layers the accuracy grew more steadily.

When, at the very last, I tested the result on the test dataset, I obtained 97% of accuracy. Because
I already verified a comparable distribution of the three datasets, I interpret this slightly lower
test accuracy as a symptom of a meta-overfitting. It could be also possible that it is instead due
to a worst image quality in the test dataset, as I could not easily check this.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![stop](http://media.gettyimages.com/photos/german-traffic-signs-picture-id459381273?s=170667a){ height=100px }
![pedestrians](http://media.gettyimages.com/photos/german-traffic-signs-picture-id459381075?s=170667a){ height=100px }
![30km/h](http://media.gettyimages.com/photos/german-traffic-signs-picture-id459381023?s=170667a){ height=100px }
![caution](http://c8.alamy.com/comp/CRDR2P/traffic-signs-achtung-unfallschwerpunkt-german-for-warning-accident-CRDR2P.jpg){ height=100px }
![right-of-way](http://bicyclegermany.com/Images/Laws/100_1607.jpg){ height=100px }

Apparentely images from web have much better quality than those of the provided dataset. The only
visible issue is that they have much more background. For this reason, I manually cropped the images
in order to keep only the relevant part.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image                                 |     Prediction                                |
|:--------------------------------------|:----------------------------------------------|
| Stop Sign                             | Stop sign                                     |
| Pedestrians                           | Pedestrians                                   |
| Speed limit (30km/h)                  | Speed limit (30km/h)                          |
| General Caution                       | General Caution                               |
| Right-of-way at the next intersection | Right-of-way at the next intersection         |


The model was able to correctly guess all 5 traffic signs, which gives an accuracy of 100%.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in cell 26 of the Ipython notebook.

In cell 31 I summarized the top-5, with predicted label as rows, true labels as columns,
probabilities as values and highlighting the maximum for each column. I report it hereafter:

| predicted                        true  | General caution  | Right-of-way at the next intersection  | Speed limit (30km/h)  | Pedestrians  | Stop        |
|:---------------------------------------|:-----------------|:---------------------------------------|:----------------------|:-------------|:------------|
| Beware of ice/snow                     | nan%             | 0.00%                                  | nan%                  | nan%         | nan%        |
| Children crossing                      | nan%             | nan%                                   | nan%                  | 0.00%        | nan%        |
| General caution                        | 100.00%          | nan%                                   | nan%                  | nan%         | nan%        |
| Keep left                              | nan%             | nan%                                   | nan%                  | nan%         | 0.00%       |
| Pedestrians                            | nan%             | 0.00%                                  | nan%                  | 99.98%       | nan%        |
| Priority road                          | nan%             | 0.00%                                  | nan%                  | nan%         | nan%        |
| Right-of-way at the next intersection  | nan%             | 100.00%                                | nan%                  | 0.00%        | nan%        |
| Road narrows on the right              | nan%             | nan%                                   | nan%                  | 0.02%        | nan%        |
| Roundabout mandatory                   | nan%             | 0.00%                                  | nan%                  | nan%         | nan%        |
| Speed limit (20km/h)                   | 0.00%            | nan%                                   | 0.00%                 | nan%         | nan%        |
| Speed limit (30km/h)                   | 0.00%            | nan%                                   | 100.00%               | nan%         | nan%        |
| Speed limit (50km/h)                   | 0.00%            | nan%                                   | 0.00%                 | nan%         | nan%        |
| Speed limit (60km/h)                   | nan%             | nan%                                   | nan%                  | nan%         | 0.00%       |
| Speed limit (70km/h)                   | nan%             | nan%                                   | 0.00%                 | nan%         | nan%        |
| Speed limit (80km/h)                   | nan%             | nan%                                   | 0.00%                 | nan%         | nan%        |
| Stop                                   | nan%             | nan%                                   | nan%                  | nan%         | 100.00%     |
| Traffic signals                        | 0.00%            | nan%                                   | nan%                  | 0.00%        | nan%        |
| Turn left ahead                        | nan%             | nan%                                   | nan%                  | nan%         | 0.00%       |
| Turn right ahead                       | nan%             | nan%                                   | nan%                  | nan%         | 0.00%       |

The network is so sure about the classification that only for the pedestrian sign returns
non-almost-zero probilities for the non-correct class. Also in this case, the top probability is
99.98%, i.e. very very high.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

In cell 33 I displayed the activation function output for the three convolutional layers on the
five images (please, remember that, thanks to the CReLU, the first 16 outputs of the first layer
are the opposite of the last 16).
The first layer highlights simple features of the images, mainly edges. In the second layer we start
to notice the ReLU effect: many feature maps are null because they are not activated by the sample
under test. When it is non-null, some rough resemblance to the original input is still visibile.
In the third layer we have all the feature maps again, but they look more like some sort of
enconding than images as in the first two layers.
