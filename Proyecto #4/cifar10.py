import numpy as np
import sys
from keras.datasets import cifar10
from keras.models import Sequential,Model
from keras.layers.core import Dense,Dropout,Flatten,Activation
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Conv2D, Convolution2D
from keras.layers import Input, AveragePooling2D, ZeroPadding2D, merge, Reshape
from keras.optimizers import SGD
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.layers.normalization import BatchNormalization
from keras import regularizers
import os
from os import path

NUM_CLASSES=10
FINAL_PATH=""
batch_size=0
epochs_number=0



def VGG16(entry_shape):
    model = Sequential()
    weight_decay = 0.0005
    model.add(Conv2D(64, (3, 3), padding='same',
    input_shape=entry_shape,kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    return model


def arch_2(entry_shape):
    model=Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=entry_shape))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))


    return model
    
def VGG19(entry_shape):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=entry_shape))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    # Add Fully Connected Layer
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    return model
def AlexNet(entry_shape):
    # Define the Model
    model = Sequential()
    # model.add(Conv2D(96, (11,11), strides=(4,4), activation='relu', padding='same', input_shape=(img_height, img_width, channel,)))
    # for original Alexnet
    model.add(Conv2D(96, (3,3), strides=(2,2), activation='relu', padding='same', input_shape=entry_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    # Local Response normalization for Original Alexnet
    model.add(BatchNormalization())

    model.add(Conv2D(256, (5,5), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))
    # Local Response normalization for Original Alexnet
    model.add(BatchNormalization())

    model.add(Conv2D(384, (3,3), activation='relu', padding='same'))
    model.add(Conv2D(384, (3,3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))
    # Local Response normalization for Original Alexnet
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(4096, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    return model    

def params_1(model):
    global epochs_number,batch_size
    epochs_number=200
    batch_size=128
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model
def params_2(model):
    global epochs_number,batch_size
    epochs_number=2
    batch_size=128
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model
def params_3(model):
    global epochs_number,batch_size
    epochs_number=200
    batch_size=128
    # Learning rate is changed to 0.001
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
def params_4(model):
    return model
def params_5(model):
    return model
def params_6(model):
    return model
def params_7(model):
    return model
def params_8(model):
    return model
def params_9(model):
    return model
def params_0(model):
    return model

def load_weights(model,weights_option,arch_name,param_name,w_option):
    FINAL_PATH=path.join(path.abspath(""),arch_name,param_name,w_option,"weights.h5")
    print(FINAL_PATH)
    if path.exists(FINAL_PATH):
        print("Ya existen los pesos para esta configuración")
        model.load_weights(FINAL_PATH)
        return True
    else:
        return False

def plotAccuracy(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    #plt.savefig("LOLO.png")


def building_training(arch_type,params_option,weights_option):
    (X_train,Y_train),(X_test,Y_test)= cifar10.load_data()
    X_train=X_train.astype('float32')
    X_test=X_test.astype('float32')
    X_train/=255.0
    X_test/=255.0
    Y_train=to_categorical(Y_train,NUM_CLASSES)
    Y_test=to_categorical(Y_test,NUM_CLASSES)

    model=arch_options[arch_type](X_train.shape[1:])
    arch_name=arch_options[arch_type].__name__
    param_name=param_options[params_option].__name__

    weights_defined=load_weights(model,weights_option,arch_name,param_name,weights_option)
    FINAL_PATH=path.join(path.abspath(""),arch_name,param_name,weights_option,"weights.h5")
    if weights_defined:
        model=param_options[params_option](model)
        scores = model.evaluate(X_test, Y_test)
    else:
        print("ENTRO A ENTRENAR EL MODELO")
        model=param_options[params_option](model)
        history=model.fit(X_train,Y_train,batch_size=batch_size,epochs=epochs_number,validation_data=(X_test,Y_test),shuffle=True)
        plotAccuracy(history)
        scores = model.evaluate(X_test, Y_test)
        if not path.exists(path.dirname(FINAL_PATH)):
            os.makedirs(path.dirname(FINAL_PATH))
        model.save_weights(FINAL_PATH)



    

    print('Loss: %.3f' % scores[0])
    print('Accuracy: %.3f' % scores[1])


arch_options={
    '1':VGG16,
    '2':arch_2,
    '3':VGG19,
    '4':AlexNet
}

param_options={
    '1':params_1,
    '2':params_2,
    '3':params_3,
    '4':params_4,
    '5':params_5,
    '6':params_6,
    '7':params_7,
    '8':params_8,
    '9':params_9,
    '0':params_0,
}


def main():
    if len(sys.argv) < 4:
        sys.exit("No se han configurado correctamente los parametros")
    arch_type=sys.argv[1]
    params_option=sys.argv[2]
    weights_option=sys.argv[3]
    building_training(arch_type,params_option,weights_option)


if __name__ == "__main__":
    main()