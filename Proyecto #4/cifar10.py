import numpy as np
import sys
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Flatten
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.utils import to_categorical

NUM_CLASSES=10



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
    preds = model.add(Dense(10, name = 'predictions'))
    #return preds
    return model


def arch_2():
    model=Sequential()
    return model
def arch_3():    
    model=Sequential()
    return model
def arch_4():
    model=Sequential()
    return model    


def building_training(arch_type,params_option,weights_option):
    (X_train,Y_train),(X_test,Y_test)= cifar10.load_data()
    X_train=X_train.astype('float32')
    X_test=X_test.astype('float32')
    X_train/=np.max(X_train)
    X_test/=np.max(X_test)
    Y_train=to_categorical(Y_train,NUM_CLASSES)
    Y_test=to_categorical(Y_test,NUM_CLASSES)

    model=arch_options[arch_type](X_train.shape[1:])

    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    model.fit(X_train,Y_train,batch_size=128,epochs=250,validation_data=(X_test,Y_test),shuffle=True)
    scores = model.evaluate(X_test / 255.0, to_categorical(Y_test))
    print('Loss: %.3f' % scores[0])
    print('Accuracy: %.3f' % scores[1])


arch_options={
    '1':VGG16,
    '2':arch_2,
    '3':arch_3,
    '4':arch_4
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