# #Define the NN architecture
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.regularizers import l1, l1_l2, l2

# It consists of 2 conv layers, 1 Dense, 1 optional Dropout, Flatten + Output layer=Dense
def first_arch(normalization=True, input_shape=(250,250,3), activation='relu', batch_size=32, dropout=0.4, learning_rate=0.0001, k_r=l2(0.)):

    model = Sequential()
    model.add(Conv2D(128, (5,5), activation=activation, kernel_regularizer=k_r, input_shape=input_shape))
    if normalization: model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(256, (5,5), activation=activation, kernel_regularizer=k_r))
    if normalization: model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Dense(2048, activation='tanh'))
    model.add(Dropout(dropout))

    model.add(Flatten())
    model.add(Dense(2, activation=('softmax')))
    adam_opt = Adam(learning_rate, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=adam_opt,loss='categorical_crossentropy',metrics=['accuracy'])

    return model

