# Dog-Breeds-Identification-APP
A Udacity Data Scientist Nanodegree Capstone Project

### Table of Contents

1. [Installation](#installation)
2. [Project Overview](#motivation)
3. [File Descriptions](#files)
4. [Instructions](#instructions)
5. [Results](#results)
6. [Conclusion](#conclusion)

## Installation <a name="installation"></a>

Beyond the Anaconda distribution of Python 3, the following packages need to be installed:
* keras
* tensorflow
* extract_bottleneck_features.py included

## Project Overview<a name="motivation"></a>

The idea of this project is to use Convolutional Neural Networks (CNNs) to build a Model/APP to process real-world, user-supplied images. Given an image of a dog, the APP will identify an estimate of the canine’s breed. If supplied an image of a human, the code will identify the resembling dog breed.

The Model/APP may provide a simple example of image content recognition and Identification. Provided with different training data, the Model/APP may be used to identificate other items, such as car, plants, animals, birds and so on.

Metrics used to measure performance of the model to be built is the dog breeds Identification accuracy.

## File Descriptions <a name="files"></a>

The project I chose is one of the Udacity suggested projects. The data (dog images) is from Udacity Deep Learning workspace which is huge and I tried but not able to download is. Thus I will not attached the data in this Repository.

- In working_directory/bottleneck_features:
  * DogResnet50Data.npz:    bottleneck features from another pre-trained CNN, I used this pre-trained CNN to build the APP

- In working_directory/data:
    * dog_names.csv:        The list of dog names extracted from the input dataset. This is for APP use as it's not efficient for the APP to read in the whole dataset and extract ths dog names list every time a user run the APP. So I saved the dog names list during my work in the jupyter notebook. It will be read when user run the APP.

- In working_directory/haarcascades:
    * haarcascade_frontalface_alt.xml:   pre-trained face detector, for function face_detector

- In working_directory/models:
    * weights.best.Resnet50.hdf5:        saved model weights with the best validation loss using Resnet50 bottleneck features
    * weights.best.from_scratch.hdf5:    saved model weights with the best validation loss for model created from scratch
    * dog_app.ipynb:                     jupyter notebook to showcase work related to the above questions. The notebooks is exploratory in building a Convolutional Neural Networks pertaining to the question showcased by the notebook title. Markdown cells & comments were used to assist in walking through the thought process for individual steps.

- In working_directory/templates:
    * *.html: HTML templates for the web app.

- In working_directory/uploads:
    * *.jpg:  For APP use, to save the images user uploads for dog breed Identification

- In working_directory:
    * extract_bottleneck_features.py:  functions to extract bottleneck features
    * app.py:                          Start the Python server for the web app and prepare visualizations(for CNN created from scratch).
    * app_Resnet50.py:                 Start the Python server for the web app and prepare visualizations(for CNN created from using Transfer Learning: Resnet50 bottleneck features).

### Instructions<a name="instructions"></a>
1. Run the following command to run the web app using CNN created from scratch
    `python app.py`

2. Run the following command to run the web app using Transfer Learning: Resnet50 bottleneck features
    `python app_Resnet50.py`

3. Go to http://0.0.0.0:3001/

## Results<a name="results"></a>

In the jupyter notebook, I followed the walk through steps provided in the notebook:
1. I built a CNN from scratch to classify dog breeds, and attained the test accuracy about 10%.
2. I built a CNN using transfer learning, pre-trained VGG-16   model as a fixed feature extractor, and attained the test accuracy about 46%.
2. I built a CNN using transfer learning, pre-trained Xception model as a fixed feature extractor, and attained the test accuracy about 85%.

When building the APP, first I tried to make it easy, just use the CNN created from scratch to build the APP infrastructure. As for this project there is not much can be done for data visualization. I just keep the APP Interface simple and intuitive. Below is the interface a user will see when he/she runs the APP.
![alt text](https://github.com/)

And the App works as expected. The Web APP asks user to upload an image and then return a message identifying an estimate of the canine’s breed. If supplied an image of a human, the APP will identify the resembling dog breed.

However, the accuracy is quite low for CNN created from scratch. For example, the below image was mis-identified to
![alt text](https://github.com/)

To improve the accuracy of the APP, I decided to use transfer learning. However the bottleneck features I used in jupyter notebook, the Xecption bottleneck features file size is too big. As I'm working on this project using Udacity Project Workspace IDE and the workspace does not allowed to upload such a huge file(3GB+), so I tried ResNet-50 bottleneck features, the file size is within acceptable range, below 100MB. I built the CNN using ResNet-50 bottleneck features in the jupyter notebook and attained the test accuracy about 82%. Although the accuracy is not as good as using Xecption bottleneck features, but it's still quite good and acceptable.
![alt text](https://github.com/)

## Conclusion<a name="conclusion"></a>

The Model/APP works as expected. However the accuracy can still be improved as the predictions of some images I tried are not accurate, just close to the actual breed. It's funny to use human images to feed the algorithm. Most human images were classified as dog breed Silky_terrier, quite intersting.

Possible improvement can be implement:
1. Add image augmentation to handle different angles, magnitude, position and partial obscurations (glasses, masks, etc).
2. Enhance image pre-process step to remove the noise in background. Such as inrelevant human or dog or patterns which may affect the prediction.
3. Add the bounding boxes of the detected faces (and detected dog) and only feed the bounding box of the face in the image to the algorithm.
