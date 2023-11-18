import cv2
import numpy as np


def projection(input_img):
    """
    投影和雷登变换
    :param input_img:
    :return:
    """
    gray_img = input_img.copy()
    if len(gray_img.shape) > 2:
        gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
    input_h, input_w = gray_img.shape


if __name__ == "__main__":
    # 图片路径
    img_sl_path = "./image/shepp-logan.png"
    # 读取图片
    img_sl = cv2.imread(img_sl_path)
