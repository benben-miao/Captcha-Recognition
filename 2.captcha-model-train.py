from keras.applications.xception import Xception,preprocess_input
from keras.layers import Input,Dense,Dropout
from keras.models import Model, load_model
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
# import keras2onnx, datetime

from sklearn.model_selection import train_test_split
from skimage import transform, io
# from scipy import misc

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import string
import glob

samples = glob.glob('./train-captcha/*.jpg')
# print(samples)
np.random.shuffle(samples)
img_size = (60, 160)

# 共有 5w 样本，4.5w 用于训练，5k 用于验证。
nb_train = 45000
train_samples = samples[:nb_train]
test_samples = samples[nb_train:]

# 10 数字 + 26 字母 = 36 分类, 此处不区分大小写。
num_let = [chr(i) for i in range(48, 58)] + [chr(i) for i in range(65, 91)]
# print(letter_list)
# ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
dig_asc = string.digits + string.ascii_uppercase
# print(dig_asc)

def buildModel():
    # CNN 适合在高宽都是偶数的情况，否则需要在边缘补齐，把全体图片都 resize 成这个尺寸(高，宽，通道)。
    input_image = Input(shape=(img_size[0], img_size[1], 3))
    # Alex Net，Google net，VGG16，VGG19，ResNet50，Xception，InceptionV3 都是由ImageNet训练而来，使用 Xception 模型用预训练，采用平均池化。
    base_model = Xception(input_tensor=input_image, weights='imagenet', include_top=False, pooling='avg')
    # 全连接层 Dense 把图片特征接上 softmax 然后 36 分类，dropout 为 0.5，因为是多分类问题，激活函数使用softmax。
    # relu 用于隐层神经元输出，sigmoid 用于隐层神经元输出，softmax 用于多分类神经网络输出，linear 用于回归神经网络输输出。
    predicts = [Dense(36, activation='softmax')(Dropout(0.5)(base_model.output)) for i in range(4)]
    model = Model(inputs=input_image, outputs=predicts)
    plot_model(model, to_file='model_xception.png', show_shapes=True, show_layer_names=True)
    return model

# io.imread 把图片转化成矩阵，
# transform.resize 重塑图片尺寸 transform.resize(io.imread(img), img_size) img_size 是自己设定的尺寸，
# ord() 函数主要用来返回对应字符的 ASCII 码，
# chr() 主要用来表示 ASCII 码对应的字符他的输入时数字，可以用十进制，也可以用十六进制。
def genCap(cap_imgs, batch_size):
    while True:
        # np.random.choice(x,y) 生成一个从 x 中抽取的随机数,维度为 y 的向量，y 为抽取次数
        batch = np.random.choice(cap_imgs, batch_size)
        x,y = [],[]
        for img in batch:
            x.append(transform.resize(io.imread(img), img_size))
            real_num = img[-8:-4]
            y_list = []
            for i in real_num:
                i = i.upper()
                if ord(i) - ord('A') >= 0:
                    y_list.append(ord(i) - ord('A') + 10)
                else:
                    y_list.append(ord(i) - ord('0'))
            y.append(y_list)
        # 把验证码标签添加到 y 列表, ord(i)-ord('a') 把对应字母转化为数字a=0, b=1, ..., z=26
        x = preprocess_input(np.array(x).astype(float))
        # 原先是 dtype=uint8 转成一个纯数字的 array
        y = np.array(y)
        yield x,[y[:,i] for i in range(4)]
        # 输出：图片 array 和四个转化成数字的字母 例如：[array([6]), array([0]), array([3]), array([24])])

'''
def onnxH5(model):
    # Convert to onnx model
    # model = load_model('model_xception.h5')
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
    keras2onnx.save_model(onnx_model, 'model_xception.onnx')
'''

def trainModel():
    model = buildModel()

    model.compile(
        optimizer='adam', 
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )

    # check_point = ModelCheckpoint(
    #     filepath='check_point.h5',
    #     save_best_only=True
    # )

    early_stop = EarlyStopping(
        monitor='val_accuracy', 
        patience=5, 
        verbose=1
    )

    model_fit = model.fit(
        genCap(train_samples, 100),
        epochs=5,
        steps_per_epoch=45000/100,
        validation_data=genCap(test_samples, 100),
        validation_steps=5000/100,
        callbacks=[early_stop, TensorBoard(log_dir='TBlogs')]
        # use_multiprocessing=True,
        # workers=5
    )
    model.save('model_xception.h5')

    # tensorboard –logdir ./TBlogs/
    # onnxH5(model)

    val_acc = model_fit.history['val_accuracy']
    np.save('val_acc.npy', val_acc)
    # val_acc = np.load('val_acc.npy')
    # plt.plot(range(len(val_acc)), val_acc, label='Model Fit Process')
    # plt.xlabel('epochs')
    # plt.ylabel('accuracy')
    # plt.legend()
    # plt.show()

if __name__ == '__main__':
    trainModel()