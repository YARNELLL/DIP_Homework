import cv2
import numpy as np


def homomorphic_filtering(input_img, d0=2, rl=1.0, rh=2.0, c=1):
    """
    同态滤波
    :param input_img: 原始图片
    :param d0: 滤波半径
    :param rl: 下限
    :param rh: 上限
    :param c: 控制坡度的锐利度
    :return: 处理后的图片
    """
    gray_img = input_img.copy()
    if len(gray_img.shape) > 2:
        gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
    input_h, input_w = gray_img.shape

    output_img = np.float64(gray_img)
    output_img = np.log(output_img + 1.0)
    output_img = output_img / np.log(256)

    output_img = np.fft.fft2(output_img)
    output_img = np.fft.fftshift(output_img)

    M, N = np.meshgrid(np.arange(-input_w // 2, input_w // 2),
                       np.arange(-input_h // 2, input_h // 2))

    D = np.sqrt(M ** 2 + N ** 2)
    Z = (rh - rl) * (1 - np.exp(-c * (D ** 2 / d0 ** 2))) + rl
    output_img = Z * output_img

    output_img = np.fft.ifftshift(output_img)
    output_img = np.fft.ifft2(output_img)

    output_img = np.abs(output_img)
    output_img = np.exp(output_img) - 1
    output_img = (output_img - output_img.min()) / (output_img.max() - output_img.min())
    output_img *= 255

    output_img = np.uint8(np.clip(output_img, 0 ,255))

    return gray_img ,output_img


if __name__ == "__main__":
    # 图片路径
    img_A_path = "./image/A.bmp"
    img_B_path = "./image/B.bmp"
    img_C_path = "./image/C.bmp"
    img_D_path = "./image/D.bmp"
    # 读取图片
    img_A = cv2.imread(img_A_path)
    img_B = cv2.imread(img_B_path)
    img_C = cv2.imread(img_C_path)
    img_D = cv2.imread(img_D_path)
    # 处理图片
    img_A_gray, img_A_filtered = homomorphic_filtering(img_A)
    img_B_gray, img_B_filtered = homomorphic_filtering(img_B)
    img_C_gray, img_C_filtered = homomorphic_filtering(img_C)
    img_D_gray, img_D_filtered = homomorphic_filtering(img_D)
    # 存储图片
    cv2.imwrite("./output/img_A_gray.png", img_A_gray)
    cv2.imwrite("./output/img_A_filtered.png", img_A_filtered)
    cv2.imwrite("./output/img_B_gray.png", img_B_gray)
    cv2.imwrite("./output/img_B_filtered.png", img_B_filtered)
    cv2.imwrite("./output/img_C_gray.png", img_C_gray)
    cv2.imwrite("./output/img_C_filtered.png", img_C_filtered)
    cv2.imwrite("./output/img_D_gray.png", img_D_gray)
    cv2.imwrite("./output/img_D_filtered.png", img_D_filtered)
