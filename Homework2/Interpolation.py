import numpy as np
import math
import cv2


def bi_linear_interpolation(input_img, output_h, output_w):
    """
    双线性插值
    :param input_img: 输入图像
    :param output_h: 输出图片的高
    :param output_w: 输出图片的宽
    :return: 双线性插值后的图像
    """
    input_h, input_w = input_img.shape

    output_img = np.zeros((output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            # 输出图片中坐标 （i，j）对应至输入图片中的最近的四个点点（x1，y1）（x2, y2），（x3， y3），(x4，y4)的均值
            temp_x = i / output_h * input_h
            temp_y = j / output_w * input_w

            x1 = math.floor(temp_x)
            y1 = math.floor(temp_y)

            u = temp_x - x1
            v = temp_y - y1

            x2 = x1
            y2 = y1 + 1

            x3 = x1 + 1
            y3 = y1

            x4 = x1 + 1
            y4 = y1 + 1

            # 防止越界
            if x4 >= input_h:
                x4 = input_h - 1
                x2 = x4
                x1 = x4 - 1
                x3 = x4 - 1
            if y4 >= input_w:
                y4 = input_w - 1
                y3 = y4
                y1 = y4 - 1
                y2 = y4 - 1

            # 插值
            output_img[i, j] = (1 - u) * (1 - v) * int(input_img[x1, y1]) + (1 - u) * v * int(
                input_img[x2, y2]) + u * (1 - v) * int(input_img[x3, y3]) + u * v * int(
                input_img[x4, y4])

    return output_img


def bi_linear_interpolation_rgb(input_img, output_h, output_w):
    """
    双线性插值
    :param input_img: 输入图像
    :param output_h: 输出图片的高
    :param output_w: 输出图片的宽
    :return: 双线性插值后的图像
    """
    input_h, input_w, channel = input_img.shape

    output_img = np.zeros((output_h, output_w, channel))

    for k in range(channel):
        for i in range(output_h):
            for j in range(output_w):
                # 输出图片中坐标 （i，j）对应至输入图片中的最近的四个点点（x1，y1）（x2, y2），（x3， y3），(x4，y4)的均值
                temp_x = i / output_h * input_h
                temp_y = j / output_w * input_w

                x1 = math.floor(temp_x)
                y1 = math.floor(temp_y)

                u = temp_x - x1
                v = temp_y - y1

                x2 = x1
                y2 = y1 + 1

                x3 = x1 + 1
                y3 = y1

                x4 = x1 + 1
                y4 = y1 + 1

                # 防止越界
                if x4 >= input_h:
                    x4 = input_h - 1
                    x2 = x4
                    x1 = x4 - 1
                    x3 = x4 - 1
                if y4 >= input_w:
                    y4 = input_w - 1
                    y3 = y4
                    y1 = y4 - 1
                    y2 = y4 - 1

                # 插值
                output_img[i, j, k] = (1 - u) * (1 - v) * int(input_img[x1, y1, k]) + (1 - u) * v * int(
                    input_img[x2, y2, k]) + u * (1 - v) * int(input_img[x3, y3, k]) + u * v * int(
                    input_img[x4, y4, k])
    return output_img


# 产生16个像素点不同的权重
def bi_cubic_weight(x):
    x = abs(x)
    if x <= 1:
        return 1 - 2.5 * (x ** 2) + 1.5 * (x ** 3)
    elif x < 2:
        return 2 - 4 * x + 2.5 * (x ** 2) - 0.5 * (x ** 3)
    else:
        return 0


def bi_cubic_interpolation(input_img, output_h, output_w):
    """
    双立方插值
    :param input_img: 输入图像
    :param output_h: 输出图像的高
    :param output_w: 输出图像的宽
    :return: 双立方插值后的图像
    """
    input_h, input_w = input_img.shape

    output_img = np.zeros((output_h, output_w))
    for i in range(output_h):
        for j in range(output_w):
            temp_x = i / output_h * input_h
            temp_y = j / output_w * input_w

            x = math.floor(temp_x)
            y = math.floor(temp_y)

            u = temp_x - x
            v = temp_y - y

            tmp = 0
            for ii in range(-1, 2):
                for jj in range(-1, 2):
                    if x + ii < 0 or y + jj < 0 or x + ii >= input_h or y + jj >= input_w:
                        continue
                    tmp += input_img[x + ii, y + jj] * bi_cubic_weight(ii - u) * bi_cubic_weight(jj - v)
            output_img[i, j] = tmp
    return output_img


def bi_cubic_interpolation_rgb(input_img, output_h, output_w):
    """
    双立方插值
    :param input_img: 输入图像
    :param output_h: 输出图像的高
    :param output_w: 输出图像的宽
    :return: 双立方插值后的图像
    """
    input_h, input_w, channel = input_img.shape

    output_img = np.zeros((output_h, output_w, channel))
    for k in range(channel):
        for i in range(output_h):
            for j in range(output_w):
                temp_x = i / output_h * input_h
                temp_y = j / output_w * input_w

                x = math.floor(temp_x)
                y = math.floor(temp_y)

                u = temp_x - x
                v = temp_y - y

                tmp = 0
                for ii in range(-1, 2):
                    for jj in range(-1, 2):
                        if x + ii < 0 or y + jj < 0 or x + ii >= input_h or y + jj >= input_w:
                            continue
                        tmp += input_img[x + ii, y + jj, k] * bi_cubic_weight(ii - u) * bi_cubic_weight(jj - v)
                output_img[i, j, k] = tmp
    return output_img


if __name__ == '__main__':
    # 读取图片
    img_depth_2x_path = './image/depth_2x.png'
    img_depth_2x = cv2.imread(img_depth_2x_path, 2)
    img_depth_2x_bi_linear_out = bi_linear_interpolation(img_depth_2x, 240, 320)
    img_depth_2x_bi_cubic_out = bi_cubic_interpolation(img_depth_2x, 240, 320)
    # 保存图片
    cv2.imwrite("depth_2x_bi_linear.png", img_depth_2x_bi_linear_out)
    cv2.imwrite("depth_2x_bi_cubic.png", img_depth_2x_bi_cubic_out)
    # 读取图片
    img_depth_4x_path = './image/depth_4x.png'
    img_depth_4x = cv2.imread(img_depth_4x_path, 2)
    img_depth_4x_bi_linear_out = bi_linear_interpolation(img_depth_4x, 240, 320)
    img_depth_4x_bi_cubic_out = bi_cubic_interpolation(img_depth_4x, 240, 320)
    # 保存图片
    cv2.imwrite("depth_4x_bi_linear.png", img_depth_4x_bi_linear_out)
    cv2.imwrite("depth_4x_bi_cubic.png", img_depth_4x_bi_cubic_out)
    # 读取图片
    img_origin_path = './image/origin.jpg'
    img_origin = cv2.imread(img_origin_path)
    img_origin_bi_linear_out = bi_linear_interpolation_rgb(img_origin, 1200, 1600)
    img_origin_bi_cubic_out = bi_cubic_interpolation_rgb(img_origin, 1200, 1600)
    # 保存图片
    cv2.imwrite("img_origin_bi_linear.png", img_origin_bi_linear_out)
    cv2.imwrite("img_origin_bi_cubic.png", img_origin_bi_cubic_out)
