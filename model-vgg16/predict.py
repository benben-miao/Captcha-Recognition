# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

from keras.layers import Input, MaxPooling2D, Dense, BatchNormalization, Flatten, Conv2D
from keras.models import Model, Sequential, load_model
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
      genCode = generator.generate_image(code_str)
      X[i] = np.array(genCode).astype('float32')/255.0

      for bat, cha in enumerate(code_str):
        y[bat][i,:] = 0
        y[bat][i, dig_asc.find(cha)] = 1
      yield X,y, dig_asc, genCode, code_str

def decode(y, dig_asc):
  y = np.argmax(np.array(y), axis=2)[:,0]
  return ','.join([dig_asc[x] for x in y])

model = load_model('model_vgg16.h5')
X, y, dig_asc, genCode, code_str = next(genCaptcha())
result = model.predict(X)
print(decode(y = result, dig_asc = dig_asc))

plt.imshow(genCode)
plt.title('Truth value: %s' %code_str, fontsize=30)
plt.show()