# coding=utf-8

from captcha.image import ImageCaptcha
from random import randint
 
def captcha_generator(captcha_num,captcha_len):
    num_let = [chr(num) for num in range(48, 58)] + [chr(lowercase) for lowercase in range(65, 91)] + [chr(uppercase) for uppercase in range(97, 123)]
    # ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
 
    for j in range(captcha_num):
        if j % 100 == 0:
            print(j)
        chars = ''
        for i in range(captcha_len):
            rand_num = randint(0, 61)
            chars += num_let[rand_num]
        image = ImageCaptcha().generate_image(chars)
        image.save('./train-captcha/' + chars + '.jpg')
 
captcha_num = 50000
captcha_len = 4
captcha_generator(captcha_num, captcha_len)