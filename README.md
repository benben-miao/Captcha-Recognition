## Captcha Recognition

### 1. Introduction
> Train Captcha Dataset -> Model Train (Keras, Tensorflow, CNN) -> Model Save -> Model Evaluation and Prediction

#### 1.1 Model Train
![train-captcha-images](https://gitee.com/benben-miao/image-cloud/raw/master/GitHub/Captcha-Recognition/train-captcha-images.png)

```python
# 验证码输入 -> 卷积层提取特征 -> 特征连接分类器 (36分类，不区分大小写)。
# Keras_weight: Alex Net，Google net，VGG16，VGG19，ResNet50，Xception，InceptionV3 都是由ImageNet训练而来。
base_model = Xception(input_tensor=input_image, weights='imagenet', include_top=False, pooling='avg')

# 全连接层把图片特征接上 softmax 然后 36 分类，dropout 为 0.5，因为是多分类问题，激活函数使用softmax。
# ReLU 用于隐层神经元输出，Sigmoid 用于隐层神经元输出，Softmax 用于多分类神经网络输出，Linear 用于回归神经网络输输出。
predicts = [Dense(36, activation='softmax')(Dropout(0.5)(base_model.output)) for i in range(4)]

model = Model(inputs=input_image, outputs=predicts)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

![train-process](https://gitee.com/benben-miao/image-cloud/raw/master/GitHub/Captcha-Recognition/train-process.png)

#### 1.2 Acc of Prediction Result
![test-multiple](https://gitee.com/benben-miao/image-cloud/raw/master/GitHub/Captcha-Recognition/test-multiple.png)

```bash
python cap-reg.py test-captcha\0Ayx.jpg test-captcha\0BDv.jpg test-captcha\0Bju.jpg
# print("Result of %s: %s" %(img, res))

# python cap-reg.py test-captcha\0Ayx.jpg test-captcha\0BDv.jpg test-captcha\0Bju.jpg 
# 2021-08-31 14:56:03.7959415 [W:onnxruntime:, execution_frame.cc:721 onnxruntime::ExecutionFrame::VerifyOutputSizes] Expected shape from model of {1,19} does not match actual shape of {1,22} for output output
# Result of test-captcha\0Ayx.jpg: 0ayx
# 2021-08-31 14:56:03.8318472 [W:onnxruntime:, execution_frame.cc:721 onnxruntime::ExecutionFrame::VerifyOutputSizes] Expected shape from model of {1,19} does not match actual shape of {1,22} for output output
# Result of test-captcha\0BDv.jpg: 0BDV
# 2021-08-31 14:56:03.8623947 [W:onnxruntime:, execution_frame.cc:721 onnxruntime::ExecutionFrame::VerifyOutputSizes] Expected shape from model of {1,19} does not match actual shape of {1,22} for output output
# Result of test-captcha\0Bju.jpg: 0BjU
```

#### 1.3 Model Diagram from Code
> CNN model plot

![CNN](https://gitee.com/benben-miao/image-cloud/raw/master/GitHub/Captcha-Recognition/model_cnn.png)

> VGG16 model plot

![VGG16](https://gitee.com/benben-miao/image-cloud/raw/master/GitHub/Captcha-Recognition/model_vgg16.png)

> Xception model plot

![Xcenption](https://gitee.com/benben-miao/image-cloud/raw/master/GitHub/Captcha-Recognition/model_xception.png)

### 2. Create Environment and Install Requirements
```bash
git clone https://github.com/benben-miao/Captcha-Recognition.git
cd Captcha-Recognition

conda create -n captcha python=3.8
conda activate captcha

pip install -r requirements.txt

python cap-reg.py test.jpg
```

### 3. Usage
![cap-rec](https://gitee.com/benben-miao/image-cloud/raw/master/GitHub/Captcha-Recognition/cap-rec.png)

```bash
# Example 1
python cap-reg.py test.jpg

# Example 2
python cap-reg.py test-captcha\0Ayx.jpg test-captcha\0BDv.jpg test-captcha\0Bju.jpg

# Result of Example2
# python cap-reg.py test-captcha\0Ayx.jpg test-captcha\0BDv.jpg test-captcha\0Bju.jpg 
# 2021-08-31 14:56:03.7959415 [W:onnxruntime:, execution_frame.cc:721 onnxruntime::ExecutionFrame::VerifyOutputSizes] Expected shape from model of {1,19} does not match actual shape of {1,22} for output output
# Result of test-captcha\0Ayx.jpg: 0ayx
# 2021-08-31 14:56:03.8318472 [W:onnxruntime:, execution_frame.cc:721 onnxruntime::ExecutionFrame::VerifyOutputSizes] Expected shape from model of {1,19} does not match actual shape of {1,22} for output output
# Result of test-captcha\0BDv.jpg: 0BDV
# 2021-08-31 14:56:03.8623947 [W:onnxruntime:, execution_frame.cc:721 onnxruntime::ExecutionFrame::VerifyOutputSizes] Expected shape from model of {1,19} does not match actual shape of {1,22} for output output
# Result of test-captcha\0Bju.jpg: 0BjU
```

