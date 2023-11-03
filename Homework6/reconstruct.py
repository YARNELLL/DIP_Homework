import cv2
import numpy as np


def gaussian_noise(input_img, mean=0, var=1e-5):
    """
    添加高斯噪音
    :param input_img:
    :param mean:
    :param var:
    :return:
    """
    input_h, input_w = input_img.shape

    output_img = np.float64(input_img)
    output_img = np.fft.fft2(output_img)
    output_img = np.fft.fftshift(output_img)

    noise = np.random.normal(mean, var ** 0.5, (input_h, input_w))

    output_img = output_img + noise

    output_img = np.fft.ifftshift(output_img)
    output_img = np.fft.ifft2(output_img)
    output_img = np.abs(output_img)
    output_img = (output_img - output_img.min()) / (output_img.max() - output_img.min())
    output_img *= 255
    output_img = np.uint8(output_img)

    noise_diff_img = input_img - output_img
    noise_diff_img = (noise_diff_img - noise_diff_img.min()) / (noise_diff_img.max() - noise_diff_img.min())
    noise_diff_img *= 255
    noise_diff_img = np.uint8(noise_diff_img)

    return output_img, noise_diff_img


def optimum_notch_filtering(input_img, d0=30, W=56):
    """
    最佳陷波滤波
    :param W:
    :param d0:
    :param input_img:
    :return:
    """
    origin_img = np.float64(input_img)
    input_h, input_w = input_img.shape

    output_img = np.float64(input_img)
    output_img = np.fft.fft2(output_img)
    output_img = np.fft.fftshift(output_img)

    h = np.zeros((input_h, input_w))
    x0 = np.floor(input_h / 2)
    y0 = np.floor(input_w / 2)
    for x in range(input_h):
        for y in range(input_w):
            d = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
            if d >= d0 + W / 2 or d <= d0 - W / 2:
                h[x][y] = 1

    h = np.multiply(output_img, h)

    eta = np.fft.ifftshift(h)
    eta = np.fft.ifft2(eta)
    eta = np.abs(eta)
    eta = (eta - eta.min()) / (eta.max() - eta.min())
    eta *= 255
    eta = np.float64(eta)

    w = np.zeros((input_h, input_w), dtype=np.float64)
    for x in range(input_h):
        for y in range(input_w):
            sum_gn = 0
            sum_g = 0
            sum_n = 0
            sum_n2 = 0
            cnt = 0
            for k in range(-1, 2):
                for j in range(-1, 2):
                    if 0 <= x + k < input_h and 0 <= y + j < input_w:
                        sum_gn += origin_img[x + k][y + j] * eta[x + k][y + j]
                        sum_g += origin_img[x + k][y + j]
                        sum_n += eta[x + k][y + j]
                        sum_n2 += np.square(eta[x + k][y + j])
                        cnt += 1
            sum_gn = sum_gn / cnt
            sum_g = sum_g / cnt
            sum_n = sum_n / cnt
            sum_n2 = sum_n2 / cnt
            w[x][y] = (sum_gn - sum_g * sum_n) / (sum_n2 - np.square(sum_n))

    output_img = output_img - w * eta

    output_img = np.fft.ifftshift(output_img)
    output_img = np.fft.ifft2(output_img)
    output_img = np.abs(output_img)
    output_img = (output_img - output_img.min()) / (output_img.max() - output_img.min())
    output_img *= 255
    output_img = np.uint8(output_img)

    optimum_diff_img = input_img - output_img
    optimum_diff_img = (optimum_diff_img - optimum_diff_img.min()) / (optimum_diff_img.max() - optimum_diff_img.min())
    optimum_diff_img *= 255
    optimum_diff_img = np.uint8(optimum_diff_img)

    return output_img, optimum_diff_img


def wiener_filtering(input_img, k=0.001, K=1e-1):
    """
    维纳滤波
    :param K:
    :param k:
    :param input_img:
    :return:
    """
    input_h, input_w = input_img.shape

    output_img = np.float64(input_img)
    output_img = np.fft.fft2(output_img)
    output_img = np.fft.fftshift(output_img)

    u, v = np.meshgrid(np.arange(-input_w // 2, input_w // 2),
                       np.arange(-input_h // 2, input_h // 2))

    h = np.exp(-k * (u ** 2 + v ** 2) ** (5 / 6))

    output_img = (np.abs(h) / (np.abs(h) + K)) / h * output_img

    output_img = np.fft.ifftshift(output_img)
    output_img = np.fft.ifft2(output_img)
    output_img = np.abs(output_img)
    output_img = (output_img - output_img.min()) / (output_img.max() - output_img.min())
    output_img *= 255
    output_img = np.uint8(output_img)

    wiener_diff_img = input_img - output_img
    wiener_diff_img = (wiener_diff_img - wiener_diff_img.min()) / (wiener_diff_img.max() - wiener_diff_img.min())
    wiener_diff_img *= 255
    wiener_diff_img = np.uint8(wiener_diff_img)

    return output_img, wiener_diff_img


if __name__ == "__main__":
    # 图片路径
    img_DIP_at_path = "./output/img_DIP_at.png"
    # 读取图片
    img_DIP_at = cv2.imread(img_DIP_at_path, 0)
    # 处理图片
    img_DIP_at_noise, img_diff_noise = gaussian_noise(img_DIP_at)
    img_DIP_at_reconstruct_wiener, img_diff_wiener = wiener_filtering(img_DIP_at_noise)
    img_DIP_at_reconstruct_optimum, img_diff_optimum = optimum_notch_filtering(img_DIP_at_noise)
    # 存储图片
    cv2.imwrite("./output/img_DIP_at_noise.png", img_DIP_at_noise)
    cv2.imwrite("./output/img_diff_noise.png", img_diff_noise)
    cv2.imwrite("./output/img_DIP_at_reconstruct_wiener.png", img_DIP_at_reconstruct_wiener)
    cv2.imwrite("./output/img_diff_wiener.png", img_diff_wiener)
    cv2.imwrite("./output/img_DIP_at_reconstruct_optimum.png", img_DIP_at_reconstruct_optimum)
    cv2.imwrite("./output/img_diff_optimum.png", img_diff_optimum)
