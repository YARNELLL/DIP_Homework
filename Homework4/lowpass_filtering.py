import cv2
import numpy as np


def ideal_lowpass_filtering(input_img, d0):
    """
    理想低通滤波
    :param d0: 滤波半径
    :param input_img: 原始图片数据
    :return: 处理后的图片
    """
    input_h, input_w = input_img.shape
    lowpass_filter = np.zeros((input_h, input_w))
    x0 = np.floor(input_h / 2)
    y0 = np.floor(input_w / 2)
    for x in range(input_h):
        for y in range(input_w):
            d = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
            if d <= d0:
                lowpass_filter[x][y] = 1
    output_img = np.multiply(input_img, lowpass_filter)
    return output_img


def gaussian_lowpass_filtering(input_img, d0):
    """
    高斯低通滤波
    :param d0: 滤波半径
    :param input_img: 原始图片数据
    :return: 处理后的图片
    """
    input_h, input_w = input_img.shape
    lowpass_filter = np.zeros((input_h, input_w))
    x0 = np.floor(input_h / 2)
    y0 = np.floor(input_w / 2)
    for x in range(input_h):
        for y in range(input_w):
            d = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
            lowpass_filter[x, y] = np.exp(-1 * d ** 2 / (2 * d0 ** 2))
    output_img = np.multiply(input_img, lowpass_filter)
    return output_img


if __name__ == "__main__":
    # 读取图片
    img_Sample_path = "./image/Sample.png"
    img_Sample = cv2.imread(img_Sample_path, 0)
    img_Sample = np.fft.fft2(img_Sample)
    img_Sample = np.fft.fftshift(img_Sample)
    # 处理图片
    img_Sample_ideal_10 = ideal_lowpass_filtering(img_Sample, 10)
    img_Sample_ideal_10 = np.fft.ifftshift(img_Sample_ideal_10)
    img_Sample_ideal_10 = np.fft.ifft2(img_Sample_ideal_10)
    img_Sample_ideal_10 = np.abs(img_Sample_ideal_10)

    img_Sample_ideal_20 = ideal_lowpass_filtering(img_Sample, 20)
    img_Sample_ideal_20 = np.fft.ifftshift(img_Sample_ideal_20)
    img_Sample_ideal_20 = np.fft.ifft2(img_Sample_ideal_20)
    img_Sample_ideal_20 = np.abs(img_Sample_ideal_20)

    img_Sample_ideal_40 = ideal_lowpass_filtering(img_Sample, 40)
    img_Sample_ideal_40 = np.fft.ifftshift(img_Sample_ideal_40)
    img_Sample_ideal_40 = np.fft.ifft2(img_Sample_ideal_40)
    img_Sample_ideal_40 = np.abs(img_Sample_ideal_40)

    img_Sample_gaussian_10 = gaussian_lowpass_filtering(img_Sample, 10)
    img_Sample_gaussian_10 = np.fft.ifftshift(img_Sample_gaussian_10)
    img_Sample_gaussian_10 = np.fft.ifft2(img_Sample_gaussian_10)
    img_Sample_gaussian_10 = np.uint8(np.abs(img_Sample_gaussian_10))

    img_Sample_gaussian_20 = gaussian_lowpass_filtering(img_Sample, 20)
    img_Sample_gaussian_20 = np.fft.ifftshift(img_Sample_gaussian_20)
    img_Sample_gaussian_20 = np.fft.ifft2(img_Sample_gaussian_20)
    img_Sample_gaussian_20 = np.uint8(np.abs(img_Sample_gaussian_20))

    img_Sample_gaussian_40 = gaussian_lowpass_filtering(img_Sample, 40)
    img_Sample_gaussian_40 = np.fft.ifftshift(img_Sample_gaussian_40)
    img_Sample_gaussian_40 = np.fft.ifft2(img_Sample_gaussian_40)
    img_Sample_gaussian_40 = np.uint8(np.abs(img_Sample_gaussian_40))
    # 存储图片
    cv2.imwrite("./output/img_Sample_ideal_10.png", img_Sample_ideal_10)
    cv2.imwrite("./output/img_Sample_ideal_20.png", img_Sample_ideal_20)
    cv2.imwrite("./output/img_Sample_ideal_40.png", img_Sample_ideal_40)
    cv2.imwrite("./output/img_Sample_gaussian_10.png", img_Sample_gaussian_10)
    cv2.imwrite("./output/img_Sample_gaussian_20.png", img_Sample_gaussian_20)
    cv2.imwrite("./output/img_Sample_gaussian_40.png", img_Sample_gaussian_40)

