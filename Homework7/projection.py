import cv2
import numpy as np


def projection(input_img):
    """
    投影和雷登变换
    :param input_img: 投影前的图片
    :return: 处理后的图片
    """
    input_h, input_w = input_img.shape
    # 正弦图参数，图像的中心
    rho_range = int(np.sqrt(np.square(input_h) + np.square(input_w)))
    theta_range = 360
    central_x = int(input_w / 2)
    central_y = int(input_h / 2)

    output_img = np.zeros((theta_range, rho_range), dtype=np.int64)

    for theta in range(theta_range):
        for x in range(-central_x, central_x + 1):
            for y in range(-central_y, central_y + 1):
                rho = (x * np.cos(theta * np.pi / theta_range) +
                       y * np.sin(theta * np.pi / theta_range))
                rho = rho_range / 2 - rho
                output_img[theta][int(rho)] += input_img[y + central_y][x + central_x]

    output_img = (output_img - output_img.min()) / (output_img.max() - output_img.min())
    output_img *= 255
    output_img = np.uint8(output_img)

    return output_img


def back_projection(input_img, output_h, output_w):
    """
    直接反投影
    :param output_w: 输出图像的宽
    :param output_h: 输出图像的长
    :param input_img: 处理前的图片
    :return: 处理后的图片
    """
    output_img = np.zeros((output_h, output_w), dtype=np.int64)
    theta_range, rho_range = input_img.shape
    central_x = int(output_w / 2)
    central_y = int(output_h / 2)

    for x in range(-central_x, central_x + 1):
        for y in range(-central_y, central_y + 1):
            for theta in range(theta_range):
                rho = (x * np.cos(theta * np.pi / theta_range) +
                       y * np.sin(theta * np.pi / theta_range))
                rho = int(rho_range / 2 - rho)
                output_img[y + central_y][x + central_x] += input_img[theta][rho]

    output_img = (output_img - output_img.min()) / (output_img.max() - output_img.min())
    output_img *= 255
    output_img = np.uint8(output_img)

    return output_img


def parallel_beam_filtered_back_projection(input_img, output_h, output_w):
    """
    使用平行射线束滤波反投影
    :param input_img: 处理前的图片
    :param output_h: 输出图像的长
    :param output_w: 输出图像的宽
    :return: 处理后的图片
    """
    output_img = np.zeros((output_h, output_w), dtype=np.float64)

    G_img = input_img.copy()
    theta_range, rho_range = input_img.shape
    filter_g = np.zeros((theta_range, rho_range), dtype=np.float64)

    central_rho = int(rho_range / 2)
    central_x = int(output_w / 2)
    central_y = int(output_h / 2)

    omiga = np.zeros(rho_range, dtype=np.float64)
    for index in range(rho_range):
        omiga[index] = np.abs(index - central_rho) * (0.54 + -0.46 * np.cos(2 * np.pi * index / (rho_range - 1)))

    for theta in range(theta_range):
        G_img_theta = G_img[theta]
        G_img_theta = np.fft.fft(G_img_theta)
        G_img_theta = np.fft.fftshift(G_img_theta)
        G_img_theta *= omiga
        G_img_theta = np.fft.ifftshift(G_img_theta)
        G_img_theta = np.fft.ifft(G_img_theta)
        filter_g[theta] = np.abs(G_img_theta)

    for x in range(-central_x, central_x + 1):
        for y in range(-central_y, central_y + 1):
            for theta in range(theta_range):
                rho = (x * np.cos(theta * np.pi / theta_range) +
                       y * np.sin(theta * np.pi / theta_range))
                rho = int(rho_range / 2 - rho)
                output_img[y + central_y][x + central_x] += filter_g[theta][rho]

    output_img = (output_img - output_img.min()) / (output_img.max() - output_img.min())
    output_img *= 255
    output_img = np.uint8(output_img)

    return output_img


if __name__ == "__main__":
    # 图片路径
    img_sl_path = "./image/shepp-logan.png"
    # 读取图片
    img_sl = cv2.imread(img_sl_path)
    # 处理图片
    # 将图片转为灰度图片
    gray_img = img_sl.copy()
    if len(gray_img.shape) > 2:
        gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
    # 进行投影
    img_sl_projection = projection(gray_img)
    # 存储图片
    cv2.imwrite("./output/img_sl_projection.png", img_sl_projection)

    # 处理图片
    img_sl_back_projection = back_projection(img_sl_projection, gray_img.shape[0], gray_img.shape[1])
    # 存储图片
    cv2.imwrite("./output/img_sl_back_projection.png", img_sl_back_projection)
    # 处理图片
    img_sl_pbf_back_projection = parallel_beam_filtered_back_projection(img_sl_projection,
                                                                        gray_img.shape[0],
                                                                        gray_img.shape[1])
    # 存储图片
    cv2.imwrite("./output/img_sl_pbf_back_projection.png", img_sl_pbf_back_projection)
