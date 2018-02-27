import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Flatten
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.utils import to_categorical


(X_train,Y_train),(X_test,Y_test)= cifar10.load_data()

NUM_CLASSES=10

X_train=X_train.astype('float32')
X_test=X_test.astype('float32')
X_train/=np.max(X_train)
X_test/=np.max(X_test)

y_train_categorical=to_categorical(Y_train,10)
y_test_categorical=to_categorical(Y_test,10)

model=Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
                  optimizer="Adam",
                  metrics=['accuracy'])


model.fit(X_train / 255.0, to_categorical(Y_train),
              batch_size=32,
              shuffle=True,
              epochs=10,
              validation_data=(X_test / 255.0, to_categorical(Y_test)))

scores = model.evaluate(X_test / 255.0, to_categorical(Y_test))

print('Loss: %.3f' % scores[0])
print('Accuracy: %.3f' % scores[1])