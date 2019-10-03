from __future__ import division
import keras
import tensorflow as tf
print( 'Using Keras version', keras.__version__)
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
        directory = '../dataset/train',
        target_size=(250, 250),
        color_mode='rgb',
        shuffle='True',
        batch_size=32,
        class_mode='categorical')

validation_generator = train_datagen.flow_from_directory(
        directory='../dataset/validation',
        target_size=(250, 250),
        color_mode='rgb',
        shuffle='True',
        batch_size=32,
        class_mode='categorical')


# #Define the NN architecture
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout

model = Sequential()
model.add(Conv2D(64, 5, 5, activation='relu', input_shape=train_generator.image_shape))
model.add(MaxPooling2D(pool_size=(3, 3)))
# model.add(BatchNormalization())


model.add(Conv2D(128, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(BatchNormalization())

model.add(Dense(2048, activation='tanh'))

model.add(Dropout(0.5))
model.add(Flatten())

model.add(Dense(2, activation=('softmax')))

#from keras.utils import plot_model
#plot_model(model, to_file='model.json', show_shapes=True)

# #Compile the NN
model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
# #Start training
model.fit_generator(
        train_generator,
        steps_per_epoch=STEP_SIZE_TRAIN,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=STEP_SIZE_VALID,
        verbose=1)


# ##Store Plots
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# #Accuracy plot
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train','val'], loc='upper left')
# plt.savefig('mnist_fnn_accuracy.pdf')
# plt.close()
# #Loss plot
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train','val'], loc='upper left')
# plt.savefig('mnist_fnn_loss.pdf')

# #Confusion Matrix
# from sklearn.metrics import classification_report,confusion_matrix
# import numpy as np
# #Compute probabilities
# Y_pred = model.predict(x_test)
# #Assign most probable label
# y_pred = np.argmax(Y_pred, axis=1)
# #Plot statistics
# print( 'Analysis of results' )
# target_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
# print(classification_report(np.argmax(y_test,axis=1), y_pred,target_names=target_names))
# print(confusion_matrix(np.argmax(y_test,axis=1), y_pred))

# #Saving model and weights
# from keras.models import model_from_json
# model_json = model.to_json()
# with open('model.json', 'w') as json_file:
#         json_file.write(model_json)
# weights_file = "weights-MNIST_"+str(score[1])+".hdf5"
# model.save_weights(weights_file,overwrite=True)

# #Loading model and weights
# #json_file = open('model.json','r')
# #model_json = json_file.read()
# #json_file.close()
# #model = model_from_json(model_json)
# #model.load_weights(weights_file)