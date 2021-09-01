# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

from keras.layers import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.utils import np_utils, plot_model
from keras.callbacks import EarlyStopping

df = pd.read_csv('data.csv')

vals = range(31)
keys = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','J','K','L','N','P','Q','R','S','T','U','V','X','Y','Z']
label_dict = dict(zip(keys, vals))

x_data = df[['v'+str(i+1) for i in range(320)]]
y_data = pd.DataFrame({'label':df['label']})
y_data['class'] = y_data['label'].apply(lambda x: label_dict[x])

X_train, X_test, Y_train, Y_test = train_test_split(x_data, y_data['class'], test_size=0.3, random_state=42)
x_train = np.array(X_train).reshape((1167, 20, 16, 1))
x_test = np.array(X_test).reshape((501, 20, 16, 1))

# 标签值进行 One-Hot encoding
n_classes = 31
y_train = np_utils.to_categorical(Y_train, n_classes)
y_val = np_utils.to_categorical(Y_test, n_classes)

def buildModel():
  input_shape = x_train[0].shape
  model = Sequential()

  # 卷积层和池化层
  model.add(Conv2D(32, kernel_size=(3, 3), input_shape=input_shape, padding='same'))
  model.add(Activation('relu'))
  model.add(Conv2D(32, kernel_size=(3, 3), padding='same'))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

  # Dropout层
  model.add(Dropout(0.25))

  model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
  model.add(Activation('relu'))
  model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

  model.add(Dropout(0.25))

  model.add(Conv2D(128, kernel_size=(3, 3), padding='same'))
  model.add(Activation('relu'))
  model.add(Conv2D(128, kernel_size=(3, 3), padding='same'))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

  model.add(Dropout(0.25))

  model.add(Flatten())

  # 全连接层
  model.add(Dense(256, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(128, activation='relu'))
  model.add(Dense(n_classes, activation='softmax'))

  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

  plot_model(model, to_file=r'model_cnn.png', show_shapes=True)
  return model

# 模型训练
def trainModel():
  model = buildModel()
  callbacks = [EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)]
  batch_size = 64
  n_epochs = 100
  history = model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epochs, \
                      verbose=1, validation_data=(x_test, y_val), callbacks=callbacks)

  model.save('model_cnn.h5')
  return history

if __name__ == '__main__':
  # 绘制验证集上的准确率曲线
  val_acc = trainModel().history['val_accuracy']
  np.save('val_acc.npy', val_acc)
  val_acc = np.load('val_acc.npy')
  plt.plot(range(len(val_acc)), val_acc, label='CNN Model')
  plt.title('Validation accuracy on verifycode dataset')
  plt.xlabel('epochs')
  plt.ylabel('accuracy')
  plt.legend()
  plt.show()