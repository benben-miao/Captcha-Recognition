import NumLetCapRec
import sys

nlc = NumLetCapRec.NumLetCap()

cap_imgs = list(sys.argv[1:])
# if len(cap_imgs) == 1:
#     with open(cap_imgs, 'rb') as f:
#         img_bytes = f.read()
#     res = nlc.recognition(img_bytes)
#     print("Result of %s: %s" %(cap_imgs, res))
# elif len(cap_imgs) > 1:
for img in cap_imgs:
    with open(img, 'rb') as f:
        img_bytes = f.read()
    res = nlc.recognition(img_bytes)
    print("Result of %s: %s" %(img, res))

# python cap-reg.py test.jpg
# python cap-reg.py test-captcha\0Ayx.jpg test-captcha\0BDv.jpg test-captcha\0Bju.jpg

# python cap-reg.py test-captcha\0Ayx.jpg test-captcha\0BDv.jpg test-captcha\0Bju.jpg 
# 2021-08-31 14:56:03.7959415 [W:onnxruntime:, execution_frame.cc:721 onnxruntime::ExecutionFrame::VerifyOutputSizes] Expected shape from model of {1,19} does not match actual shape of {1,22} for output output
# Result of test-captcha\0Ayx.jpg: 0ayx
# 2021-08-31 14:56:03.8318472 [W:onnxruntime:, execution_frame.cc:721 onnxruntime::ExecutionFrame::VerifyOutputSizes] Expected shape from model of {1,19} does not match actual shape of {1,22} for output output
# Result of test-captcha\0BDv.jpg: 0BDV
# 2021-08-31 14:56:03.8623947 [W:onnxruntime:, execution_frame.cc:721 onnxruntime::ExecutionFrame::VerifyOutputSizes] Expected shape from model of {1,19} does not match actual shape of {1,22} for output output
# Result of test-captcha\0Bju.jpg: 0BjU