import cv2
import numpy as np


def symbiotic_matrix(input_img):
    """
    计算共生矩阵，并且返回参数指标
    :param input_img:   输入的图像
    :return: 共生矩阵以及描绘子
    """
    k = 256
    input_h, input_w = input_img.shape
    output_img = np.zeros((k, k), dtype=np.uint64)
    for x in range(input_h):
        for y in range(input_w - 1):
            output_img[input_img[x][y], input_img[x][y + 1]] += 1

    output_img = np.log(1 + output_img)

    output_img = (output_img - output_img.min()) / (output_img.max() - output_img.min())
    output_img *= 255
    output_img = np.uint8(output_img)

    total = output_img.sum()
    p = output_img / total
    m_r = np.sum([i * np.sum(p[i, :]) for i in range(k)])
    m_c = np.sum([j * np.sum(p[:, j]) for j in range(k)])
    sigma_r = np.sum([(i - m_r) ** 2 * np.sum(p[i, :]) for i in range(k)])
    sigma_c = np.sum([(j - m_c) ** 2 * np.sum(p[:, j]) for j in range(k)])

    max_probability = np.max(p)
    correlation = 0
    for i in range(k):
        for j in range(k):
            correlation += (i - m_r) * (j - m_c) * p[i][j] / (sigma_c * sigma_r)
    contrast = 0
    for i in range(k):
        for j in range(k):
            contrast += (i - j) ** 2 * p[i][j]
    consistency = 0
    for i in range(k):
        for j in range(k):
            consistency += p[i][j] ** 2
    homogeneity = 0
    for i in range(k):
        for j in range(k):
            homogeneity += p[i][j] / (1 + np.abs(i - j))
    entropy = 0
    for i in range(k):
        for j in range(k):
            entropy += -p[i][j] * np.log2(p[i][j] + 1e-7)

    return output_img, max_probability, correlation, contrast, consistency, homogeneity, entropy


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
    (img_G1_symbiotic_matrix, img_G1_max_probability, img_G1_correlation, img_G1_contrast, img_G1_consistency,
     img_G1_homogeneity, img_G1_entropy) = symbiotic_matrix(img_G1)
    (img_G2_symbiotic_matrix, img_G2_max_probability, img_G2_correlation, img_G2_contrast, img_G2_consistency,
     img_G2_homogeneity, img_G2_entropy) = symbiotic_matrix(img_G2)
    (img_G3_symbiotic_matrix, img_G3_max_probability, img_G3_correlation, img_G3_contrast, img_G3_consistency,
     img_G3_homogeneity, img_G3_entropy) = symbiotic_matrix(img_G3)
    # 保存图片，输出描绘子
    cv2.imwrite("./output/img_G1_symbiotic_matrix.png", img_G1_symbiotic_matrix)
    cv2.imwrite("./output/img_G2_symbiotic_matrix.png", img_G2_symbiotic_matrix)
    cv2.imwrite("./output/img_G3_symbiotic_matrix.png", img_G3_symbiotic_matrix)
    print("最大概率，\tG1：{:.10f}，\tG2：{:.10f}，\tG3：{:.10f}".format(
        img_G1_max_probability, img_G2_max_probability, img_G3_max_probability))
    print("相关，\tG1：{:.10f}，\tG2：{:.10f}，\tG3：{:.10f}".format(
        img_G1_correlation, img_G2_correlation, img_G3_correlation))
    print("对比度，\tG1：{:.10f}，\tG2：{:.10f}，\tG3：{:.10f}".format(
        img_G1_contrast, img_G2_contrast, img_G3_contrast))
    print("一致性，\tG1：{:.10f}，\tG2：{:.10f}，\tG3：{:.10f}".format(
        img_G1_consistency, img_G2_consistency, img_G3_consistency))
    print("同质性，\tG1：{:.10f}，\tG2：{:.10f}，\tG3：{:.10f}".format(
        img_G1_homogeneity, img_G2_homogeneity, img_G3_homogeneity))
    print("熵，\tG1：{:.10f}，\tG2：{:.10f}，\tG3：{:.10f}".format(
        img_G1_entropy, img_G2_entropy, img_G3_entropy))
