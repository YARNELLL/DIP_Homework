import cv2
import numpy as np


def enhance(input_img, b_length):
    """
    图像增强
    :param b_length: 结构元的边长
    :param input_img: 处理前的图片
    :return: 通过开操作和闭操作增强后的图片
    """
    input_h, input_w = input_img.shape
    padding_length = int(np.floor(b_length / 2))
    output_img = np.zeros((input_h + padding_length * 2, input_w + padding_length * 2), dtype=np.uint8)
    b_matrix = np.ones((b_length, b_length), dtype=np.uint8)

    for x in range(input_h):
        for y in range(input_w):
            output_img[padding_length + x][padding_length + y] = input_img[x][y]

    output_img = opening(output_img, b_matrix, padding_length)
    output_img = closing(output_img, b_matrix, padding_length)

    return output_img[padding_length:-padding_length][padding_length:-padding_length]


def erosion(a_matrix, b_matrix, padding_length):
    """
    腐蚀
    :param a_matrix:
    :param b_matrix:
    :param padding_length:
    :return:
    """
    shift_img = [np.roll(np.roll(a_matrix, i, axis=0), j, axis=1)
                 for i in range(-padding_length, padding_length + 1)
                 for j in range(-padding_length, padding_length + 1)]
    mul_img = [shift_img[i] * b_matrix.flatten()[i]
               for i in range(b_matrix.shape[0] * b_matrix.shape[1])]
    output_img = np.minimum.reduce(mul_img)
    return output_img


def dilation(a_matrix, b_matrix, padding_length):
    """
    膨胀
    :param a_matrix:
    :param b_matrix:
    :param padding_length:
    :return:
    """
    shift_img = [np.roll(np.roll(a_matrix, i, axis=0), j, axis=1)
                 for i in range(-padding_length, padding_length + 1)
                 for j in range(-padding_length, padding_length + 1)]
    mul_img = [shift_img[i] * b_matrix.flatten()[i]
               for i in range(b_matrix.shape[0] * b_matrix.shape[1])]
    output_img = np.maximum.reduce(mul_img)
    return output_img


def opening(a_matrix, b_matrix, padding_length):
    """
    开操作
    :param a_matrix:
    :param b_matrix:
    :param padding_length:
    :return:
    """
    output_img = erosion(a_matrix, b_matrix, padding_length)
    output_img = dilation(output_img, b_matrix, padding_length)
    return output_img


def closing(a_matrix, b_matrix, padding_length):
    """
    闭操作
    :param a_matrix:
    :param b_matrix:
    :param padding_length:
    :return:
    """
    output_img = dilation(a_matrix, b_matrix, padding_length)
    output_img = erosion(output_img, b_matrix, padding_length)
    return output_img


if __name__ == "__main__":
    # 图片路径
    img_fingerprint_path = "./image/fingerprint.tif"
    # 读取图片
    img_fingerprint = cv2.imread(img_fingerprint_path, 2)
    # 处理图片
    img_fingerprint_enhanced = enhance(img_fingerprint, 3)
    # 存储图片
    cv2.imwrite("./output/img_fingerprint_enhanced.tif", img_fingerprint_enhanced)
