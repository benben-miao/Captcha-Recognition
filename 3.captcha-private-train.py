from keras.applications.xception import Xception,preprocess_input
from keras.layers import Input,Dense,Dropout
from keras.models import Model
from skimage import transform, io
# from scipy import misc
import numpy as np
import glob

img_size = (60, 160)
input_image = Input(shape=(img_size[0],img_size[1],3))
base_model = Xception(input_tensor=input_image, weights=None, include_top=False, pooling='avg')
predicts = [Dense(36, activation='softmax')(Dropout(0.5)(base_model.output)) for i in range(4)]

model = Model(inputs=input_image, outputs=predicts)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.load_weights('CaptchaNumberLetter.h5')

def data_generator(data, batch_size):
    while True:
        batch = np.random.choice(data, batch_size)
        x,y = [],[]
        for img in batch:
            im = io.imread(img)
            im = im[:, :, :3]
            im = transform.resize(im, img_size)
            x.append(im)
            real_num = img[-8:-4]

            y_list = []
            for i in real_num:
                if ord(i)-ord('A') >= 0:
                    y_list.append(ord(i)-ord('A')+10)
                else:
                    y_list.append(ord(i)-ord('0'))

            y.append(y_list)
        x = preprocess_input(np.array(x).astype(float))
        y = np.array(y)
        yield x,[y[:,i] for i in range(4)]

# 获取指定目录下的所有图片
samples = glob.glob('./dianxin-captcha/*.jpg')
np.random.shuffle(samples)

nb_train = 450
train_samples = samples[:nb_train]
test_samples = samples[nb_train:]
# print(train_samples)

# Continue training
model.fit_generator(data_generator(train_samples, 30), steps_per_epoch=15, epochs=7, validation_data=data_generator(test_samples, 10), validation_steps=5)
model.save('CaptchaDianXin.h5')