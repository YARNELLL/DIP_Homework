import numpy as np
import cv2


def histogram_equalization(input_img):
    """
    直方图均衡化
    :param input_img: 输入的图像
    :return: 处理完后的图像
    """
    input_h, input_w = input_img.shape
    output_img = np.zeros((input_h, input_w))
    # 统计灰度级的数目
    histogram = np.zeros(256, dtype=np.uint16)
    for i in range(input_h):
        for j in range(input_w):
            gray_data = input_img[i][j]
            histogram[gray_data] += 1
    # 映射函数
    input_img_size = input_w * input_h
    lut = np.zeros(256, dtype=np.float16)
    cnt = 0
    for i in range(256):
        cnt += histogram[i]
        lut[i] = cnt / input_img_size * 255
    # 图像变换
    for i in range(input_h):
        for j in range(input_w):
            output_img[i][j] = lut[input_img[i][j]]
    return output_img


def local_region_stretch_histogram_equalization(input_img):
    """
    自适应局部区域伸展直方图均衡化
    :param input_img: 输入的图片
    :return: 处理后的图片
    """
    input_h, input_w = input_img.shape
    output_img = np.zeros((input_h, input_w))
    # 统计灰度级的数目
    histogram = np.zeros(256, dtype=np.uint16)
    pixel_cnt_0_to_70 = 0
    pixel_cnt_70_to_130 = 0
    pixel_cnt_130_to_255 = 0
    for i in range(input_h):
        for j in range(input_w):
            gray_data = input_img[i][j]
            histogram[gray_data] += 1
            if gray_data < 70:
                pixel_cnt_0_to_70 += 1
            elif gray_data < 130:
                pixel_cnt_70_to_130 += 1
            else:
                pixel_cnt_130_to_255 += 1
    # 映射函数
    lut = np.zeros(256, dtype=np.float16)
    cnt = 0
    if pixel_cnt_0_to_70 != 0:
        for i in range(70):
            cnt += histogram[i]
            lut[i] = cnt / pixel_cnt_0_to_70 * 69

    if pixel_cnt_70_to_130 != 0:
        cnt = 0
        for i in range(70, 130):
            cnt += histogram[i]
            lut[i] = 70 + cnt / pixel_cnt_70_to_130 * 59
    if pixel_cnt_130_to_255 != 0:
        cnt = 0
        for i in range(130, 256):
            cnt += histogram[i]
            lut[i] = 130 + cnt / pixel_cnt_130_to_255 * 126
    # 图像变换
    for i in range(input_h):
        for j in range(input_w):
            output_img[i][j] = lut[input_img[i][j]]
    return output_img


if __name__ == '__main__':
    # 读取图片
    img_scene_view_path = './image/sceneview.jpg'
    img_tungsten_path = './image/tungsten_original.jpg'
    img_scene_view = cv2.imread(img_scene_view_path, 0)
    img_tungsten = cv2.imread(img_tungsten_path, 0)
    # 处理图片
    output_img_scene_view = histogram_equalization(img_scene_view)
    output_img_tungsten = histogram_equalization(img_tungsten)
    output_img_scene_view_local = local_region_stretch_histogram_equalization(img_scene_view)
    output_img_tungsten_local = local_region_stretch_histogram_equalization(img_tungsten)
    # 保存图片
    cv2.imwrite("./output/output_img_scene_view.png", output_img_scene_view)
    cv2.imwrite("./output/output_img_tungsten.png", output_img_tungsten)
    cv2.imwrite("./output/output_img_scene_view_local.png", output_img_scene_view_local)
    cv2.imwrite("./output/output_img_tungsten_local.png", output_img_tungsten_local)
