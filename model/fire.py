from __future__ import division
import keras
import tensorflow as tf
print( 'Using Keras version', keras.__version__)
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l1, l1_l2, l2
from keras.callbacks import TensorBoard

from model import first_arch 
import getpass

import os
import shutil
import argparse

parser = argparse.ArgumentParser(description='Fire CNN model')
parser.add_argument('--save_dir', type=str, help='directory to save results')
args = parser.parse_args()
save_dir = args.save_dir

dataset_dir = '/home/nct01/{}/.keras/datasets/dataset'.format(getpass.getuser())
train_datagen = ImageDataGenerator(
        rotation_range=20,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
        directory = dataset_dir + '/train',
        target_size=(250, 250),
        color_mode='rgb',
        shuffle='True',
        batch_size=32,
        class_mode='categorical')

valid_generator = train_datagen.flow_from_directory(
        directory=dataset_dir + '/valid',
        target_size=(250, 250),
        color_mode='rgb',
        shuffle='True',
        batch_size=32,
        class_mode='categorical')

image_shape = train_generator.image_shape

kwargs = {
    "first_layer":
        {
            "filters": 128,
            "kernel_size": (3,3),
            "activation": 'relu',
            "kernel_regularizer": l2(0.)
        },
    "second_layer":
        {
            "filters": 256,
            "kernel_size": (3,3),
            "activation": 'relu',
            "kernel_regularizer": l2(0.)
        },
    "dense_layer":
        {
            "units": 2048,
            "activation": 'tanh'
            # "kernel_regularizer":l2
        },
    "dropout": 0.1,
    "pool_size": (2,2)
}

model = first_arch(input_shape=image_shape, normalization=False,**kwargs)

# from keras.utils import plot_model
#     plot_model(model, to_file='model.json', show_shapes=True)

#Callbacks
tbCallBack = keras.callbacks.TensorBoard(log_dir='../logs', histogram_freq=0, write_graph=True, write_images=True)

# #Compile the NN
model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
# #Start training
history = model.fit_generator(
        train_generator,
        steps_per_epoch=STEP_SIZE_TRAIN,
        epochs=50,
        validation_data=valid_generator,
        validation_steps=STEP_SIZE_VALID,
        verbose=1,
        callbacks=[tbCallBack])


##Store Plots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#Accuracy plot
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','valid'], loc='upper left')
plt.savefig(save_dir + '/acc_fire.pdf')
plt.close()

#Loss plot
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','val'], loc='upper left')
plt.savefig(save_dir + "/loss_fire.pdf")

# Confusion Matrix
# from sklearn.metrics import classification_report,confusion_matrix
# import numpy as np
# Compute probabilities
# Y_pred = model.predict(x_test)
# Assign most probable label
# y_pred = np.argmax(Y_pred, axis=1)
# Plot statistics
# print( 'Analysis of results' )
# target_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
# print(classification_report(np.argmax(y_test,axis=1), y_pred,target_names=target_names))
# print(confusion_matrix(np.argmax(y_test,axis=1), y_pred))

#Saving model and weights
from keras.models import model_from_json
model_json = model.to_json()
with open(save_dir + '/model.json', 'w') as json_file:
        json_file.write(model_json)
weights_file = "weights-fire_"+str(score[1])+".hdf5"
model.save_weights(save_dir + weights_file,overwrite=True)

#Loading model and weights
#json_file = open('model.json','r')
#model_json = json_file.read()
#json_file.close()
#model = model_from_json(model_json)
#model.load_weights(weights_file)
