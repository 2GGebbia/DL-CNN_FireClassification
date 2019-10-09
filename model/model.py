# #Define the NN architecture
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout

# It consists of 2 conv layers, 1 Dense, 1 optional Dropout, Flatten + Output layer=Dense
def first_arch(input_shape, normalization=True, **kwargs):

    first_layer_dict = kwargs['first_layer']
    model = Sequential()
    model.add(Conv2D(**first_layer_dict, input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(3,3)))
    if normalization: model.add(BatchNormalization())  


    second_layer_dict = kwargs['second_layer']
    model.add(Conv2D(**second_layer_dict))
    model.add(MaxPooling2D(pool_size=(3,3)))
    if normalization: model.add(BatchNormalization())

    dense_layer_dict = kwargs['dense_layer']
    model.add(Dense(2048, activation='tanh'))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(2, activation=('softmax')))

    return model