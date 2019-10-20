from __future__ import division
import keras
import tensorflow as tf
print( 'Using Keras version', keras.__version__)
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l1, l1_l2, l2
from keras.callbacks import EarlyStopping

from model import first_arch
import getpass
from random import uniform

dataset_dir = '/home/nct01/{}/.keras/datasets/dataset'.format(getpass.getuser())
train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        shear_range=0.2,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
)

valid_datagen = ImageDataGenerator(
        rescale=1./255)

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

kwargs = {
        "first_layer_conv":
        {
                "filters": 128,
                "kernel_size": (5,5),
                "activation": 'relu',
                "kernel_regularizer": l2(0.)
        },
        "second_layer_conv":
        {
                "filters": 256,
                "kernel_size": (5,5),
                "activation": 'relu',
                "kernel_regularizer": l2(0.)
        },
        "dense_layer":
        {
                "units": 2048,
                "activation": 'tanh',
                "kernel_regularizer": l2(0.)
        },
        "dropout": 0,
        "pool_size": (2,2)
}

for dropout in [0.3, 0.4, 0.5, 0.6]:
        for learning_rate in [0.001, 0.0001, 0.00001]:

                print("\n\n\n")
                print(15*"=")
                print(15*"=")
                print(learning_rate, dropout)
                print(15*"=")
                print(15*"=")
                print("\n\n\n")
                
                kwargs["dropout"] = dropout

                model = first_arch(input_shape=image_shape, normalization=False,**kwargs)

                # Compile the NN
                sgd = keras.optimizers.SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
                model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])


                # Callbacks
                es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)


                STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
                STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
                # Start training
                history = model.fit_generator(
                        train_generator,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        epochs=15,
                        validation_data=valid_generator,
                        validation_steps=STEP_SIZE_VALID,
                        verbose=1,
                        callbacks=[eg])

                # Store Plots
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt

                # Accuracy plot
                plt.plot(history.history['acc'])
                plt.plot(history.history['val_acc'])
                plt.title('model accuracy')
                plt.ylabel('accuracy')
                plt.xlabel('epoch')
                plt.legend(['train','valid'], loc='upper left')
                plt.savefig('acc_fire_dr{}lr_{}.pdf'.format(dropout, learning_rate))
                plt.close()

                # Loss plot
                plt.plot(history.history['loss'])
                plt.plot(history.history['val_loss'])
                plt.title('model loss')
                plt.ylabel('loss')
                plt.xlabel('epoch')
                plt.legend(['train','val'], loc='upper left')
                plt.savefig("loss_fire_dr{}lr_{}.pdf".format(dropout, learning_rate))
                plt.close()

                
                Saving model and weights
                from keras.models import model_from_json
                model_json = model.to_json()
                with open('model.json', 'w') as json_file:
                        json_file.write(model_json)
                weights_file = "weights-fire.hdf5"
                model.save_weights(weights_file,overwrite=True)
