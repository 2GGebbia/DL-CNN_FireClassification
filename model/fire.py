from __future__ import division
import keras
import tensorflow as tf
print( 'Using Keras version', keras.__version__)
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l1, l1_l2, l2
from keras.callbacks import TensorBoard
from new_model import create_model
import getpass
import os

import matplotlib
# Store Plots
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from keras.models import model_from_json

# SLURM JOB ID 
job_id = os.environ['SLURM_JOB_ID']

# DATASET DIRECTORY
dataset_dir = '/home/nct01/{}/.keras/datasets/dataset'.format(getpass.getuser())

# Data Image Generators for Train and Validation
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

## MANUALLY GRID SEARCH 

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
for learning_rate in [10**-2, 10**-3, 10**-4, 10**-5]:
        for dropout in [i/10 for i in range(1,8)]:

                print("\n\n\n")
                print(15*"=")
                print(15*"=")
                print(learning_rate, dropout)
                print(15*"=")
                print(15*"=")
                print("\n\n\n")

                model = create_model(dropout, learning_rate)

                # Start training
                history = model.fit_generator(
                        train_generator,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        epochs=20,
                        validation_data=valid_generator,
                        validation_steps=STEP_SIZE_VALID,
                        verbose=1)



                # Accuracy plot
                plt.plot(history.history['acc'])
                plt.plot(history.history['val_acc'])
                plt.title('model accuracy')
                plt.ylabel('accuracy')
                plt.xlabel('epoch')
                plt.legend(['train','valid'], loc='upper left')
                plt.savefig('results/acc_fire_drop_{}_lr_{}_job_{}.pdf'.format(dropout, learning_rate, job_id))
                plt.close()

                # Loss plot
                plt.plot(history.history['loss'])
                plt.plot(history.history['val_loss'])
                plt.title('model loss')
                plt.ylabel('loss')
                plt.xlabel('epoch')
                plt.legend(['train','valid'], loc='upper left')
                plt.savefig('results/loss_fire_drop_{}_lr_{}_job_{}.pdf'.format(dropout, learning_rate, job_id))

                # Saving model and weights
                model_json = model.to_json()
                with open('results/model_drop_{}_lr_{}_job_{}.pdf'.format(dropout, learning_rate, job_id), 'w') as json_file:
                        json_file.write(model_json)
                # model.save_weights('results/weights_drop_{}_lr_{}_job_{}.hdf5'.format(dropout, learning_rate, job_id) ,overwrite=True)

#Loading model and weights
#json_file = open('model.json','r')
#model_json = json_file.read()
#json_file.close()
#model = model_from_json(model_json)
#model.load_weights(weights_file)
