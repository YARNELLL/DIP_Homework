import math
import cv2
import numpy as np


def fourier_transformation(input_img):
    """
    二维离散傅里叶变换
    :param input_img: 输入的图像
    :return: 处理完后的图像
    """
    input_h, input_w = input_img.shape
    output_img = np.zeros((input_h, input_w), dtype=complex)

    for u in range(input_h):
        for v in range(input_w):
            temp = 0
            for x in range(input_h):
                for y in range(input_w):
                    temp += input_img[x, y] * np.exp(-1j * 2 * math.pi * (u * x / input_h + v * y / input_w))
            output_img[u, v] = temp

    output_img_log = np.log(1 + np.abs(output_img))
    return output_img, output_img_log


def inverse_fourier_transformation(input_img):
    """
    二维离散傅里叶逆变换
    :param input_img: 输入的图像
    :return: 处理完后的图像
    """
    input_h, input_w = input_img.shape
    output_img = np.zeros((input_h, input_w))

    for x in range(input_h):
        for y in range(input_w):
            temp = 0
            for u in range(input_h):
                for v in range(input_w):
                    temp += input_img[u, v] * np.exp(1j * 2 * math.pi * (u * x / input_h + v * y / input_w))
            output_img[x, y] = temp / (input_h * input_w)

    return output_img


if __name__ == '__main__':
    # 读取图片
    img_Lena_path = "./image/Lena.png"
    img_Sample_path = "./image/Sample.png"
    img_Lena = cv2.imread(img_Lena_path, 0)
    img_Sample = cv2.imread(img_Sample_path, 0)
    # 处理图片
    output_img_Lena, output_img_Lena_log = fourier_transformation(img_Lena)
    output_img_Sample, output_img_Sample_log = fourier_transformation(img_Sample)
    output_img_Lena_inverse = inverse_fourier_transformation(output_img_Lena)
    output_img_Sample_inverse = inverse_fourier_transformation(output_img_Lena_inverse)
    # 存储图片
    cv2.imwrite("./output/output_img_Lena.png", np.abs(output_img_Lena))
    cv2.imwrite("./output/output_img_Sample.png", np.abs(output_img_Sample))
    cv2.imwrite("./output/output_img_Lena_log.png", output_img_Lena_log)
    cv2.imwrite("./output/output_img_Sample_log.png", output_img_Sample_log)
    cv2.imwrite("./output/output_img_Lena_inverse.png", output_img_Lena_inverse)
    cv2.imwrite("./output/output_img_Sample_inverse.png", output_img_Sample_inverse)
