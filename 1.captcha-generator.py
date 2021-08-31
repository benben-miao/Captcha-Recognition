# coding=utf-8

from captcha.image import ImageCaptcha
from random import randint
 
def captcha_generator(captcha_num,captcha_len):
    # 10数字 + 26大写字母 + 26小写字母
    list = [chr(num) for num in range(48, 58)] + [chr(lowercase) for lowercase in range(65, 91)] + [chr(uppercase) for uppercase in range(97, 123)]
 
    for j in range(captcha_num):
        if j % 100 == 0:
            print(j)
        chars = ''
        for i in range(captcha_len):
            rand_num = randint(0, 61)
            chars += list[rand_num]
        image = ImageCaptcha().generate_image(chars)
        image.save('./train-captcha/' + chars + '.jpg')
 
captcha_num = 50000
captcha_len = 4
captcha_generator(captcha_num,captcha_len)