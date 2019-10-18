from __future__ import division
import keras
import tensorflow as tf
print( 'Using Keras version', keras.__version__)
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l1, l1_l2, l2
from keras.callbacks import TensorBoard
from new_model import first_arch
import getpass

import os


# job_id = os.environ['SLURM_JOB_ID']
dataset_dir = '/home/nct01/{}/.keras/datasets/dataset'.format(getpass.getuser())
# dataset_dir = 'dataset'
train_datagen = ImageDataGenerator(
        rotation_range=20,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)

valid_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        directory = dataset_dir + '/train',
        target_size=(250, 250),
        color_mode='rgb',
        shuffle='True',
        batch_size=32,
        class_mode='categorical')

valid_generator = valid_datagen.flow_from_directory(
        directory=dataset_dir + '/valid',
        target_size=(250, 250),
        color_mode='rgb',
        shuffle='True',
        batch_size=32,
        class_mode='categorical')

image_shape = train_generator.image_shape
print(image_shape)
params_dict = {
    "learning_rate": [10^-4, 10^-5, 10^-6],
    "dropout": [0.1, 0.3, 0.5, 0.7],
    "batch_size": [16, 32, 64],
    "activation": ['relu', 'elu'],
    "k_r": [l1(0.), l2(0.)]
}

model = first_arch(input_shape=image_shape, normalization=False)

#Callbacks
tbCallBack = keras.callbacks.TensorBoard(log_dir='../logs', histogram_freq=0, write_graph=True, write_images=True)

# #Compile the NN

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
# #Start training
history = model.fit_generator(
        train_generator,
        steps_per_epoch=STEP_SIZE_TRAIN,
        epochs=5,
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
plt.savefig('../results/acc_fire_{}.pdf'.format(job_id))
plt.close()

#Loss plot
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','val'], loc='upper left')
plt.savefig("../results/loss_fire_{}.pdf".format(job_id))

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
with open( '../results/model_{}.json'.format(job_id), 'w') as json_file:
        json_file.write(model_json)
weights_file = "weights-fire_"+str(score[1])+".hdf5"
model.save_weights('../results/{}_{}'.format(job_id, weights_file) ,overwrite=True)

#Loading model and weights
#json_file = open('model.json','r')
#model_json = json_file.read()
#json_file.close()
#model = model_from_json(model_json)
#model.load_weights(weights_file)
