from keras.models import load_model
import numpy as np
from skimage import transform, io
# from scipy import misc
from keras.applications.xception import preprocess_input
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import glob
 
img_size = (60, 160)
model = load_model('./CaptchaNumberLetter.h5')
letter_list = [chr(i) for i in range(48,58)] + [chr(i) for i in range(65,91)]
 
def data_generator_test(data, n): # 样本生成器，节省内存
    while True:
        batch = np.array([data[n]])
        x,y = [],[]
        for img in batch:
            im = io.imread(img)
            im = im[:, :, :3]
            im = transform.resize(im, img_size)
            x.append(im) # 读取resize图片,再存进x列表
            y_list = []
 
            real_num = img[-8:-4]
            for i in real_num:
                if ord(i)-ord('A') >= 0:
                    y_list.append(ord(i)-ord('A')+10)
                else:
                    y_list.append(ord(i)-ord('0'))
 
            y.append(y_list) # 把验证码标签添加到y列表,ord(i)-ord('a')把对应字母转化为数字a=0，b=1……z=26
            # print('real_1:',img[-8:-4])
        x = preprocess_input(np.array(x).astype(float)) # 原先是 dtype=uint8 转成一个纯数字的 array
        y = np.array(y)
 
        yield x,[y[:,i] for i in range(4)]
 
def predict2(n):
    x,y = next(data_generator_test(test_samples, n))
    z = model.predict(x)
    z = np.array([i.argmax(axis=1) for i in z]).T
    result = z.tolist()
    v = []
    for i in range(len(result)):
        for j in result[i]:
            v.append(letter_list[j])
 
    # 输出测试结果
    str = ''
    for i in v:
        str += i
 
    real = ''
    for i in y:
        for j in i:
            real += letter_list[j]
    return (str,real)
 
test_samples = glob.glob(r'test-captcha/*.jpg')
 
n = 0
n_right = 0
for i in range(len(test_samples)):
    n += 1
    print('~~~~~~~~~~~~~%d~~~~~~~~~'%(n))
    predict,real = predict2(i)
 
    if real == predict:
        n_right += 1
    else:
        print('real:', real)
        print('predict:',predict)
        image = mpimg.imread(test_samples[i])
        plt.axis('off')
        plt.imshow(image)
        plt.show()
 
print(n,n_right,n_right/n)