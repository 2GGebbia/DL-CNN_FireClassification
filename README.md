# DL-CNN_FireClassification

The aim of this project is to provide a CNN neural network able to classify whether an image contains image or not. In order to do so we created our own dataset and implemented two different architectures.

## Dataset Adquisition

The dataset used for this project was manually searched from google images using the [google_images_download](https://github.com/hardikvasa/google-images-download) API such that given a keyword, the size of the picture and the number of pictures to be downloaded it gets from google images all the specified photos. 
In order to obtain a representative dataset we downloaded from different environments related with fire (forests, kitchen, buildings, candles) in distinct lighty days (sunny, cloud, night).

From this tool we also downloaded nonfire images of nature, people talking, buildings, picnic photos and several more in order to make it the less biased possible.

We also obtained around 500 fire photos from a Fire Detection repository. 

All images are stored in the `raw_dataset` folder splitted by two subfolders (fire and nonfire). Using the python script `processing_data.py` located at misc and the specified percentages splitting, it creates a `dataset` folder with the correct structure in order to load it later using `flow_from_directory`.


### ***First Architecture***

CNN made of:\
- 2 conv layers with Max Pooling
- 1 Dense layer of 2048 neurons and tanh activation function
- Dropout layer
- Flatten
- Dense of 2 neurons with softmax function (Output layer)

After several tries, we decided to fix some hyperparameters as the number of filters of the convolutional layer, the activation functions, and the batch size.

Hyperparameters to study: **dropout** and **learning_rate**

A GridSearch was used during the executions in order to explore the best configuration.

### ***Second Architecture***

CNN made of:\
- 2 conv layers with Max Pooling
- 1 Dense layer of 2048 neurons and tanh activation function
- Dropout layer
- Flatten
- Dense of 2 neurons with softmax function (Output layer)

After several tries, we decided to fix some hyperparameters as the number of filters of the convolutional layer, the activation functions, and the batch size.

Hyperparameters to study: **dropout** and **learning_rate**

A RandomSearch was used during the executions in order to explore the best configuration.