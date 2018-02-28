import numpy as np
import sys
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Flatten
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import os
from os import path

NUM_CLASSES=10
FINAL_PATH=""
batch_size=0
epochs_number=0



def VGG16(entry_shape):
    model=Sequential()
    # BLOCK 1 
    model.add(Conv2D(64, (3, 3),    activation='relu',  padding='same',     name = 'BLOCK1_Conv1',  input_shape=entry_shape))
    model.add(Conv2D(64, (3, 3),    activation='relu',  padding='same',     name = 'BLOCK1_Conv2'))
    model.add(MaxPooling2D((2,2),   strides = (2,2),                        name = 'BLOCK1_Pool1'))

    # BLOCK 2
    model.add(Conv2D(128, (3, 3),   activation='relu',  padding='same',     name = 'BLOCK2_Conv1'))
    model.add(Conv2D(128, (3, 3),   activation='relu',  padding='same',     name = 'BLOCK2_Conv2'))
    model.add(MaxPooling2D((2,2),   strides = (2,2),                        name = 'BLOCK2_Pool2'))

    # BLOCK 3
    model.add(Conv2D(256, (3,3),    activation = 'relu', padding = 'same',  name = 'block3_conv1'))
    model.add(Conv2D(256, (3,3),    activation = 'relu', padding = 'same',  name = 'block3_conv2'))
    model.add(Conv2D(256, (3,3),    activation = 'relu', padding = 'same',  name = 'block3_conv3'))
    model.add(MaxPooling2D((2,2),   strides = (2,2),                        name = 'block3_pool'))

    # BLOCK 4
    model.add(Conv2D(512, (3,3),    activation = 'relu', padding = 'same',  name = 'block4_conv1'))
    model.add(Conv2D(512, (3,3),    activation = 'relu', padding = 'same',  name = 'block4_conv2'))
    model.add(Conv2D(512, (3,3),    activation = 'relu', padding = 'same',  name = 'block4_conv3'))
    model.add(MaxPooling2D((2,2),   strides = (2,2),                        name = 'block4_pool'))
    
    # BLOCK 5
    model.add(Conv2D(512, (3,3),    activation = 'relu', padding = 'same', name = 'block5_conv1'))
    model.add(Conv2D(512, (3,3),    activation = 'relu', padding = 'same', name = 'block5_conv2'))
    model.add(Conv2D(512, (3,3),    activation = 'relu', padding = 'same', name = 'block5_conv3'))
    model.add(MaxPooling2D((2,2),   strides = (2,2),                        name = 'block5_pool'))
    
    # Classification BLOCK
    model.add(Flatten(name = 'flatten'))
    model.add(Dense(4096, activation = 'relu', name = 'fc1'))
    model.add(Dense(4096, activation = 'relu', name = 'fc2'))
    model.add(Dense(10, name = 'predictions'))
    #return preds
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
def arch_3():    
    model=Sequential()
    return model
def arch_4():
    model=Sequential()
    return model    

def params_1(model):
    global epochs_number,batch_size
    epochs_number=200
    batch_size=128
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model
def params_2(model):
    return model
def params_3(model):
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
        #plotAccuracy(history)
        scores = model.evaluate(X_test, Y_test)
        if not path.exists(path.dirname(FINAL_PATH)):
            os.makedirs(path.dirname(FINAL_PATH))
        model.save_weights(FINAL_PATH)



    

    print('Loss: %.3f' % scores[0])
    print('Accuracy: %.3f' % scores[1])


arch_options={
    '1':VGG16,
    '2':arch_2,
    '3':arch_3,
    '4':arch_4
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