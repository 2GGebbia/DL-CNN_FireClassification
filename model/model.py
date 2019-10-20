# #Define the NN architecture
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout


# It consists of 2 conv layers, 1 Dense, 1 optional Dropout, Flatten + Output layer=Dense
def first_arch(input_shape, normalization=True, **kwargs):

    model = Sequential()
    
    first_layer_dict = kwargs['first_layer_conv']
    model.add(Conv2D(**first_layer_dict, input_shape=input_shape))
    if normalization: model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=kwargs["pool_size"]))

    second_layer_dict = kwargs['second_layer_conv']
    model.add(Conv2D(**second_layer_dict))
    if normalization: model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=kwargs["pool_size"]))

    dense_layer_dict = kwargs['dense_layer']
    model.add(Dense(**dense_layer_dict))
    model.add(Dropout(kwargs["dropout"]))

    model.add(Flatten())
    model.add(Dense(2, activation=('softmax')))

    return model


# It consists of 2 conv layers, 2 Dense, 1 optional Dropout, Flatten + Output layer=Dense
def second_arch(input_shape, normalization=True, **kwargs):

    first_layer_conv_dict = kwargs['first_layer_conv']
    model = Sequential()
    model.add(Conv2D(**first_layer_conv_dict, input_shape=input_shape))
    if normalization: model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=kwargs["pool_size"]))

    second_layer_conv_dict = kwargs['second_layer_conv']
    model.add(Conv2D(**second_layer_conv_dict))
    if normalization: model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=kwargs["pool_size"]))

    first_layer_dense_dict = kwargs['first_layer_dense']
    model.add(Dense(**first_layer_dense_dict))
    model.add(Dropout(kwargs["dropout"]))

    second_layer_dense_dict = kwargs['first_layer_dense']
    model.add(Dense(**second_layer_dense_dict))
    model.add(Dropout(kwargs["dropout"]))

    model.add(Flatten())
    model.add(Dense(2, activation=('softmax')))

    return model