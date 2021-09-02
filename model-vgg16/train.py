# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

from keras.layers import Input, MaxPooling2D, Dense, BatchNormalization, Flatten, Conv2D
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils.vis_utils import plot_model

from captcha.image import ImageCaptcha
from PIL import Image
import string, random
import keras2onnx, datetime
import numpy as np
import matplotlib.pyplot as plt

def randomCode():
  dig_asc = string.digits + string.ascii_uppercase
  # cap_str = ''.join([random.choice(dig_asc) for i in range(4)])
  cap_str = ''.join(random.sample(dig_asc, 4))
  return dig_asc, cap_str

def genCap(height=80, width=170, batch_size=32, n_class=36):
  X = np.zeros((batch_size, height, width, 3), dtype=np.uint8)
  y = [np.zeros((batch_size, n_class), dtype=np.uint8) for i in range(4)]

  cap_gen = ImageCaptcha(height=height, width=width)
  while True:
    for i in range(batch_size):
      dig_asc, cap_str = randomCode()
      # cap_img = cap_gen.generate_image(cap_str)
      # cap_img.save()
      X[i] = cap_gen.generate_image(cap_str)
      for pos, char in enumerate(cap_str):
        y[pos][i, :] = 0
        y[pos][i, dig_asc.find(char)] = 1
      yield X,y

def buildModel():
  height, width, n_class = 80, 170, 36
  input_tensor = Input(shape=(height, width, 3))
  x = input_tensor
  # VGG16 神经网络层
  for i in range(4):
    x = Conv2D(filters=32*2**i, kernel_size=(3, 3), activation='relu')(x)
    x = Conv2D(filters=32*2**i, kernel_size=(3, 3), activation='relu')(x)
    x = BatchNormalization(axis=3)(x)
    x = MaxPooling2D((2, 2))(x)
  x = Flatten()(x)
  x = [Dense(n_class, activation='softmax', name='D%d'%(n+1))(x) for n in range(4)]
  model = Model(inputs=input_tensor, outputs=x)
  plot_model(model, to_file='model_vgg16.png', show_shapes=True, show_layer_names=True)
  return model

def onnxH5(model):
    # Convert to onnx model
    # model = load_model('model_vgg16.h5')
    onnx_model = keras2onnx.convert_keras(model, model.name)

    # Add metadata to the ONNX model.
    meta = onnx_model.metadata_props.add()
    meta.key = "date"
    meta.value = datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    meta = onnx_model.metadata_props.add()
    meta.key = "author"
    meta.value = 'benben-miao'
    onnx_model.doc_string = 'Model of Captcha'
    onnx_model.model_version = 1
    keras2onnx.save_model(onnx_model, 'model_vgg16.onnx')

def trainModel():
  model = buildModel()

  model.compile(
    loss='categorical_crossentropy',
    optimizer='adadelta',
    metrics=['accuracy']
  )

  check_point = ModelCheckpoint(
    filepath='model_vgg16.h5',
    save_best_only=True
  )

  model.fit_generator(
    genCap(),
    epochs=5,
    steps_per_epoch=10000,
    validation_data=genCap(),
    validation_steps=1000,
    callbacks=[check_point, TensorBoard(log_dir='TB_logs')]
    # tensorboard –logdir ./TB_logs/
  )
  
  # onnxH5(model)

if __name__ == "__main__":
  trainModel()