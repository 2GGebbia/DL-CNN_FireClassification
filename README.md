# DL-CNN_FireClassification
### Marc Asenjo & Gian Carlo Gebbia

The aim of this project is to provide a CNN neural network able to classify whether an image contains a fire or not. In order to do so we created our own dataset and implemented two different general Convolutional Network Architectures.

## Dataset Adquisition

The dataset used for this project was manually searched from google images using the [google_images_download](https://github.com/hardikvasa/google-images-download) API such that given a keyword, the size of the picture and the number of pictures to be downloaded it gets from google images all the specified photos. 
In order to obtain a representative dataset we downloaded from different environments related with fire (forests, kitchen, buildings, candles) in distinct lighty days (sunny, cloud, night).

From this tool we also downloaded nonfire images of nature, people talking, buildings, picnic photos and several more in order to make it as un-biased as possible.

We also obtained around 500 fire photos from a Fire Detection repository (https://github.com/OlafenwaMoses/FireNET).

All images are stored in the `raw_dataset` folder splitted by two subfolders (fire and nonfire). Using the python script `processing_data.py` located at misc and the specified percentages splitting, it creates a `dataset` folder with the correct structure in order to load it later using `flow_from_directory` in Keras.

Finally, the structure chosen for the development of the project is a 85% as training data and 15% as validation data. Because the analysis we are going to make is a comparison between hyperparameter configurations, the validation dataset is the one used for this puropose, and no test set is required. The final structure that we end up obtaining with all given numbers for each data partition and class is shown in the next table:

|              | Fire | No Fire | Total |
|--------------|-----:|--------:|------:|
|  **Training**|  759 |   707   |  1466 |
|**Validation**|  134 |   124   |  258  |
|    **Total** |  893 |   831   |  1724 |


### ***First Architecture***

CNN made of:
- 2 conv layers with Max Pooling
- **1 Dense layer** of 2048 neurons and tanh activation function
- Dropout applied to the Dense Layer output
- Dense layer of 2 neurons with softmax function (Output layer)
- Use of 'categorical cross_entropy' loss function

After several tests for many of the hyperparameters, we decided to fix some hyperparameters as the number of filters of the convolutional layer, the activation functions, and the batch size.

Hyperparameters studied in more depth: **dropout** and **learning_rate**.

A **GridSearch** was used during the executions in order to explore the best configuration.

### ***Second Architecture***

CNN made of:
- 2 conv layers with Max Pooling
- **2 Dense layers** of 512 / 1024 neurons and tanh activation function
- Dropout applied to the Dense Layer output
- Dense layer of 2 neurons with softmax function (Output layer)
- Use of 'categorical cross_entropy' loss function

After several tests for many of the hyperparameters, we decided to fix some hyperparameters as the number of filters of the convolutional layer, the activation functions, and the batch size.

Hyperparameters studied in more depth: **dropout** and **learning_rate**.

A **RandomSearch** was used during the executions in order to explore the best configuration.