from keras.applications.xception import Xception,preprocess_input
from keras.layers import Input,Dense,Dropout
from keras.models import Model
# from scipy import misc
from skimage import transform, io
import numpy as np
import glob

samples = glob.glob('./train-captcha/*.jpg')
np.random.shuffle(samples)

# 共有 5w 样本，4.5w 用于训练，5k 用于验证。
nb_train = 45000
train_samples = samples[:nb_train]
test_samples = samples[nb_train:]

# 10 数字 + 26 字母 = 36 分类, 此处不区分大小写。
letter_list = [chr(i) for i in range(48, 58)] + [chr(i) for i in range(65, 91)]

# CNN 适合在高宽都是偶数的情况，否则需要在边缘补齐，把全体图片都 resize 成这个尺寸(高，宽，通道)。
img_size = (60, 160)
input_image = Input(shape=(img_size[0],img_size[1],3))

# 验证码输入 -> 卷积层提取特征 -> 特征连接分类器 (36分类，不区分大小写)。
# Keras_weight: Alex Net，Google net，VGG16，VGG19，ResNet50，Xception，InceptionV3 都是由ImageNet训练而来，
# 这里使用 Xception 模型用预训练，采用平均池化。
base_model = Xception(input_tensor=input_image, weights='imagenet', include_top=False, pooling='avg')

# 全连接层把图片特征接上 softmax 然后 36 分类，dropout 为 0.5，因为是多分类问题，激活函数使用softmax。
# ReLU 用于隐层神经元输出，
# Sigmoid 用于隐层神经元输出，
# Softmax 用于多分类神经网络输出，
# Linear 用于回归神经网络输输出。
predicts = [Dense(36, activation='softmax')(Dropout(0.5)(base_model.output)) for i in range(4)]

model = Model(inputs=input_image, outputs=predicts)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# io.imread 把图片转化成矩阵，
# transform.resize 重塑图片尺寸 transform.resize(io.imread(img), img_size) img_size 是自己设定的尺寸，
# ord() 函数主要用来返回对应字符的 ASCII 码，
# chr() 主要用来表示 ASCII 码对应的字符他的输入时数字，可以用十进制，也可以用十六进制。
def data_generator(data, batch_size): #样本生成器，节省内存
    while True:
        # np.random.choice(x,y) 生成一个从 x 中抽取的随机数,维度为 y 的向量，y 为抽取次数
        batch = np.random.choice(data, batch_size)
        x,y = [],[]
        for img in batch:
            # 读取 resize 图片,再存进 x 列表
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
 
model.fit_generator(data_generator(train_samples, 100), steps_per_epoch=450, epochs=5, validation_data=data_generator(test_samples, 100), validation_steps=100)
# 参数：generator生成器函数,
# samples_per_epoch，每个 epoch 以经过模型的样本数达到 samples_per_epoch 时，记一个 epoch 结束
# step_per_epoch: 整数，当生成器返回 step_per_epoch 次数据是记一个 epoch 结束，执行下一个 epoch
# epochs: 整数，数据迭代的轮数
# validation_data 三种形式之一，生成器，类（inputs,targets）的元组，或者（inputs,targets，sample_weights）的元祖
# 若 validation_data 为生成器，validation_steps 参数代表验证集生成器返回次数
# class_weight：规定类别权重的字典，将类别映射为权重，常用于处理样本不均衡问题。
# sample_weight：权值的 numpy array，用于在训练时调整损失函数（仅用于训练）。可以传递一个1D的与样本等长的向量用于对样本进行1对1的加权，或者在面对时序数据时，传递一个的形式为（samples，sequence_length）的矩阵来为每个时间步上的样本赋不同的权。这种情况下请确定在编译模型时添加了sample_weight_mode='temporal'。
# workers：最大进程数
# max_q_size：生成器队列的最大容量
# pickle_safe: 若为真，则使用基于进程的线程。由于该实现依赖多进程，不能传递 non picklable（无法被 pickle 序列化）的参数到生成器中，因为无法轻易将它们传入子进程中。
# initial_epoch: 从该参数指定的epoch开始训练，在继续之前的训练时有用。
 
# 保存模型
model.save('CaptchaNumberLetter.h5')