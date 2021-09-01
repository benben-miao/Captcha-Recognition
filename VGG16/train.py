# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

from keras.layers import Input, MaxPooling2D, Dense, BatchNormalization, Flatten, Conv2D
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils.vis_utils import plot_model

from captcha.image import ImageCaptcha
from PIL import Image
import string, random
import numpy as np
import matplotlib.pyplot as plt

def randomCode():
  dig_asc = string.digits + string.ascii_uppercase
  # print(dig_asc)
  code = ''.join(random.sample(dig_asc, 4))
  # print(code)
  return code, dig_asc

def genCaptcha(height=80, width=170, batch_size=32, n_class=36):
  X = np.zeros((batch_size, height, width, 3), dtype=np.float)
  y = [np.zeros((batch_size, n_class), dtype=np.uint8) for i in range(4)]

  generator = ImageCaptcha(height=height, width=width)
  while 1:
    for i in range(batch_size):
      code_str, dig_asc = randomCode()
      # code_img = generator.generate_image(code_str)
      # code_img.save()
      X[i] = np.array(generator.generate_image(code_str)).astype('float32')/255.0
      for bat, cha in enumerate(code_str):
        y[bat][i,:] = 0
        y[bat][i, dig_asc.find(cha)] = 1
      yield X,y

def trainBreakModel():
  h, w, nclass = 80, 170, 36
  input_tensor = Input(shape=(h, w, 3))
  x = input_tensor
  # VGG16 神经网络层
  for i in range(4):
    x = Conv2D(filters=32*2**i, kernel_size=(3, 3), activation='relu')(x)
    x = Conv2D(filters=32*2**i, kernel_size=(3, 3), activation='relu')(x)
    x = BatchNormalization(axis=3)(x)
    x = MaxPooling2D((2, 2))(x)
  x = Flatten()(x)
  x = [Dense(nclass, activation='softmax', name='D%d'%(n+1))(x) for n in range(4)]
  model = Model(inputs=input_tensor, outputs=x)
  return model
# plot_model(trainBreakModel(), to_file='model_plot.png', show_shapes=True, show_layer_names=True)

def train():
  model = trainBreakModel()

  check_point = ModelCheckpoint(
    filepath='check_point.h5',
    save_best_only=True
  )

  model.compile(
    loss='categorical_crossentropy',
    optimizer='adadelta',
    metrics=['accuracy']
  )

  model.fit_generator(
    genCaptcha(),
    epochs=10,
    steps_per_epoch=50,
    validation_data=genCaptcha(),
    validation_steps=10,
    callbacks=[check_point, TensorBoard(log_dir='TB_logs')]
  )

if __name__ == "__main__":
  train()