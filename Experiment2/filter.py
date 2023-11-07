import cv2
import numpy as np


def alpha_trimmed_mean_filter(input_img, m=5, n=5, d=5):
    """
    修正的阿尔法均值滤波器
    :param d: 截取大小
    :param n: 窗口边长
    :param m: 窗口边长
    :param input_img: 输入的图像
    :return: 处理后的图像
    """
    input_h, input_w = input_img.shape
    output_img = np.zeros((input_h, input_w), dtype=np.float64)
    padding_length_m = int(np.floor(m / 2))
    padding_length_n = int(np.floor(n / 2))
    padding_img = np.zeros((input_h + 2 * padding_length_m, input_w + 2 * padding_length_n))
    for x in range(input_h):
        for y in range(input_w):
            padding_img[x + padding_length_m][y + padding_length_n] = input_img[x][y]

    for x in range(input_h):
        for y in range(input_w):
            window_g = []
            for k in range(padding_length_m * 2 + 1):
                for j in range(padding_length_n * 2 + 1):
                    window_g.append(padding_img[x + k][y + j])
            window_g = np.sort(window_g)
            window_g = window_g[int(np.ceil(d / 2)): -int(np.floor(d / 2))]
            output_img[x][y] = np.sum(window_g) / (m * n - d)

    output_img = (output_img - output_img.min()) / (output_img.max() - output_img.min())
    output_img *= 255
    output_img = np.uint8(output_img)
    return output_img


def adaptive_median_filter(input_img, s0=3, smax=7):
    """
    自适应中值滤波器
    :param smax: 最大的窗口大小
    :param s0: 初始窗口大小
    :param input_img: 原始图像
    :return: 处理后的图像
    """

    input_h, input_w = input_img.shape
    output_img = np.zeros((input_h, input_w))
    max_padding_length_m = int(np.floor(smax / 2))
    max_padding_length_n = int(np.floor(smax / 2))
    padding_img = np.zeros((input_h + 2 * max_padding_length_m, input_w + 2 * max_padding_length_n))
    for x in range(input_h):
        for y in range(input_w):
            padding_img[x + max_padding_length_m][y + max_padding_length_n] = input_img[x][y]

    for x in range(input_h):
        for y in range(input_w):
            cur_s = s0
            while cur_s <= smax:
                window_g = []
                cur_padding_length_m = int(np.floor(cur_s / 2))
                cur_padding_length_n = int(np.floor(cur_s / 2))
                for k in range(-cur_padding_length_m, cur_padding_length_m + 1):
                    for j in range(-cur_padding_length_n, cur_padding_length_n + 1):
                        window_g.append(padding_img[x + max_padding_length_m + k][y + max_padding_length_n + j])
                z_max = np.max(window_g)
                z_med = np.median(window_g)
                z_min = np.min(window_g)
                z_xy = padding_img[x + max_padding_length_m][y + max_padding_length_n]
                if z_med - z_min > 0 > z_med - z_max:
                    if z_xy - z_min > 0 > z_xy - z_max:
                        output_img[x][y] = z_xy
                        break
                    else:
                        output_img[x][y] = z_med
                        break
                else:
                    cur_s += 2
                    if cur_s > smax:
                        output_img[x][y] = z_med
                        break

    return output_img


if __name__ == "__main__":
    # 图片路径
    img_A_Gauss_path = "./image/A_Gauss.bmp"
    img_A_SP_05_path = "./image/A_SP_05.bmp"
    img_A_SP_1_path = "./image/A_SP_1.bmp"
    img_A_SP_2_path = "./image/A_SP_2.bmp"
    img_A_SP_5_path = "./image/A_SP_5.bmp"
    # 读取图片
    img_A_Gauss = cv2.imread(img_A_Gauss_path, 0)
    img_A_SP_05 = cv2.imread(img_A_SP_05_path, 0)
    img_A_SP_1 = cv2.imread(img_A_SP_1_path, 0)
    img_A_SP_2 = cv2.imread(img_A_SP_2_path, 0)
    img_A_SP_5 = cv2.imread(img_A_SP_5_path, 0)
    # 处理图片
    img_A_Gauss_alpha = alpha_trimmed_mean_filter(img_A_Gauss, d=2)
    img_A_SP_05_alpha = alpha_trimmed_mean_filter(img_A_SP_05)
    img_A_SP_1_alpha = alpha_trimmed_mean_filter(img_A_SP_1, d=10)
    img_A_SP_2_alpha = alpha_trimmed_mean_filter(img_A_SP_2, d=15)
    img_A_SP_5_alpha = alpha_trimmed_mean_filter(img_A_SP_5, d=20)
    img_A_Gauss_median = adaptive_median_filter(img_A_Gauss)
    img_A_SP_05_median = adaptive_median_filter(img_A_SP_05)
    img_A_SP_1_median = adaptive_median_filter(img_A_SP_1)
    img_A_SP_2_median = adaptive_median_filter(img_A_SP_2)
    img_A_SP_5_median = adaptive_median_filter(img_A_SP_5)
    # 存储图片
    cv2.imwrite("./output/img_A_Gauss_alpha.png", img_A_Gauss_alpha)
    cv2.imwrite("./output/img_A_SP_05_alpha.png", img_A_SP_05_alpha)
    cv2.imwrite("./output/img_A_SP_1_alpha.png", img_A_SP_1_alpha)
    cv2.imwrite("./output/img_A_SP_2_alpha.png", img_A_SP_2_alpha)
    cv2.imwrite("./output/img_A_SP_5_alpha.png", img_A_SP_5_alpha)
    cv2.imwrite("./output/img_A_Gauss_median.png", img_A_Gauss_median)
    cv2.imwrite("./output/img_A_SP_05_median.png", img_A_SP_05_median)
    cv2.imwrite("./output/img_A_SP_1_median.png", img_A_SP_1_median)
    cv2.imwrite("./output/img_A_SP_2_median.png", img_A_SP_2_median)
    cv2.imwrite("./output/img_A_SP_5_median.png", img_A_SP_5_median)
