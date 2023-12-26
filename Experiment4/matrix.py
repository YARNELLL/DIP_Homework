import cv2
import numpy as np


def symbiotic_matrix(input_img):
    """
    计算共生矩阵，并且返回参数指标
    :param input_img:   输入的图像
    :return: 共生矩阵
    """
    input_h, input_w = input_img.shape
    output_img = np.zeros((256, 256), dtype=np.uint64)
    for x in range(input_h):
        for y in range(input_w - 1):
            output_img[input_img[x][y], input_img[x][y + 1]] += 1

    output_img = (output_img - output_img.min()) / (output_img.max() - output_img.min())
    output_img *= 255
    output_img = np.uint8(output_img)

    return output_img


if __name__ == "__main__":
    # 图片路径
    img_G1_path = "./image/G1.bmp"
    img_G2_path = "./image/G2.bmp"
    img_G3_path = "./image/G3.bmp"
    img_M1_path = "./image/M1.bmp"
    img_M2_path = "./image/M2.bmp"
    img_M3_path = "./image/M3.bmp"
    img_M4_path = "./image/M4.bmp"
    img_M5_path = "./image/M5.bmp"
    img_M6_path = "./image/M6.bmp"
    # 读取图片
    img_G1 = cv2.imread(img_G1_path, 0)
    img_G2 = cv2.imread(img_G2_path, 0)
    img_G3 = cv2.imread(img_G3_path, 0)
    img_M1 = cv2.imread(img_M1_path, 0)
    img_M2 = cv2.imread(img_M2_path, 0)
    img_M3 = cv2.imread(img_M3_path, 0)
    img_M4 = cv2.imread(img_M4_path, 0)
    img_M5 = cv2.imread(img_M5_path, 0)
    img_M6 = cv2.imread(img_M6_path, 0)
    # 处理图片
    img_G1_symbiotic_matrix = symbiotic_matrix(img_G1)
    img_G2_symbiotic_matrix = symbiotic_matrix(img_G2)
    img_G3_symbiotic_matrix = symbiotic_matrix(img_G3)
    # 保存图片
    cv2.imwrite("./output/img_G1_symbiotic_matrix.png", img_G1_symbiotic_matrix)
    cv2.imwrite("./output/img_G2_symbiotic_matrix.png", img_G2_symbiotic_matrix)
    cv2.imwrite("./output/img_G3_symbiotic_matrix.png", img_G3_symbiotic_matrix)
