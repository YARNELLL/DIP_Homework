import cv2
import numpy as np


def atmospheric_turbulence_model(input_img, k=0.001):
    """
    大气湍流模型
    :param k:
    :param input_img:
    :return:
    """
    gray_img = input_img.copy()
    if len(gray_img.shape) > 2:
        gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
    input_h, input_w = gray_img.shape

    output_img = np.float64(gray_img)
    output_img = np.fft.fft2(output_img)
    output_img = np.fft.fftshift(output_img)

    u, v = np.meshgrid(np.arange(-input_w // 2, input_w // 2),
                       np.arange(-input_h // 2, input_h // 2))

    h = np.exp(-k * (u ** 2 + v ** 2) ** (5 / 6))

    output_img = output_img * h

    output_img = np.fft.ifftshift(output_img)
    output_img = np.fft.ifft2(output_img)
    output_img = np.abs(output_img)
    output_img = (output_img - output_img.min()) / (output_img.max() - output_img.min())
    output_img *= 255
    output_img = np.uint8(output_img)

    return output_img


def motion_model(input_img, a=0.05, b=0.05, T=1):
    """
    运动模糊模型
    :param input_img:
    :param a:
    :param b:
    :param T:
    :return:
    """
    gray_img = input_img.copy()
    if len(gray_img.shape) > 2:
        gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
    input_h, input_w = gray_img.shape

    output_img = np.float64(gray_img)
    output_img = np.fft.fft2(output_img)
    output_img = np.fft.fftshift(output_img)

    u, v = np.meshgrid(np.arange(-input_w // 2, input_w // 2),
                       np.arange(-input_h // 2, input_h // 2))

    h = (T / (np.pi * (u * a + v * b + 1e-6)) * np.sin(np.pi * (u * a + v * b + 1e-6)) *
         np.exp(-1j * np.pi * (u * a + v * b + 1e-6)))

    output_img = output_img * h

    output_img = np.fft.ifftshift(output_img)
    output_img = np.fft.ifft2(output_img)
    output_img = np.abs(output_img)
    output_img = (output_img - output_img.min()) / (output_img.max() - output_img.min())
    output_img *= 255
    output_img = np.uint8(output_img)

    return output_img


if __name__ == "__main__":
    # 图片路径
    img_DIP_path = "./image/DIP.bmp"
    # 读取图片
    img_DIP = cv2.imread(img_DIP_path)
    # 处理图片
    img_DIP_at = atmospheric_turbulence_model(img_DIP)
    img_DIP_motion = motion_model(img_DIP)
    # 存储图片
    cv2.imwrite("./output/img_DIP_at.png", img_DIP_at)
    cv2.imwrite("./output/img_DIP_motion.png",img_DIP_motion)
