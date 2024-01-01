import cv2
import numpy as np


def symbiotic_matrix(input_img):
    """
    计算共生矩阵，并且返回描绘子指标
    :param input_img:   输入的图像
    :return: 共生矩阵以及描绘子
    """
    k = 256
    input_h, input_w = input_img.shape
    output_img = np.zeros((k, k), dtype=np.uint64)
    for x in range(input_h):
        for y in range(input_w - 1):
            output_img[input_img[x][y], input_img[x][y + 1]] += 1

    # 共生矩阵
    s_matrix = output_img

    output_img = np.log(1 + output_img)

    output_img = (output_img - output_img.min()) / (output_img.max() - output_img.min())
    output_img *= 255
    output_img = np.uint8(output_img)

    total = s_matrix.sum()
    p = s_matrix / total
    m_r = np.sum([i * np.sum(p[i, :]) for i in range(k)])
    m_c = np.sum([j * np.sum(p[:, j]) for j in range(k)])
    sigma_r = np.sqrt(np.sum([(i - m_r) ** 2 * np.sum(p[i, :]) for i in range(k)]))
    sigma_c = np.sqrt(np.sum([(j - m_c) ** 2 * np.sum(p[:, j]) for j in range(k)]))

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


def invariant_moment(input_img):
    """
    计算不变矩
    :param input_img:
    :return: 返回不变矩计算结果
    """
    m_00 = compute_m_pq(input_img, 0, 0)
    m_01 = compute_m_pq(input_img, 0, 1)
    m_10 = compute_m_pq(input_img, 1, 0)
    x_bar = m_10 / m_00
    y_bar = m_01 / m_00
    mu_00 = compute_mu_pq(input_img, 0, 0, x_bar, y_bar)

    eta_20 = compute_eta_pq(input_img, 2, 0, x_bar, y_bar, mu_00)
    eta_02 = compute_eta_pq(input_img, 0, 2, x_bar, y_bar, mu_00)
    phi_1 = eta_20 + eta_02

    eta_11 = compute_eta_pq(input_img, 1, 1, x_bar, y_bar, mu_00)
    phi_2 = (eta_20 - eta_02) ** 2 + 4 * eta_11 ** 2

    eta_30 = compute_eta_pq(input_img, 3, 0, x_bar, y_bar, mu_00)
    eta_12 = compute_eta_pq(input_img, 1, 2, x_bar, y_bar, mu_00)
    eta_21 = compute_eta_pq(input_img, 2, 1, x_bar, y_bar, mu_00)
    eta_03 = compute_eta_pq(input_img, 0, 3, x_bar, y_bar, mu_00)
    phi_3 = (eta_30 - 3 * eta_12) ** 2 + (3 * eta_21 - eta_03) ** 2

    phi_4 = (eta_30 + eta_12) ** 2 + (eta_21 + eta_03) ** 2

    phi_5 = (eta_30 - 3 * eta_12) * (eta_30 + eta_12) * ((eta_30 + eta_12) ** 2 - 3 * (eta_21 + eta_03) ** 2) + \
            (3 * eta_21 - eta_03) * (eta_21 + eta_03) * (3 * (eta_30 + eta_12) ** 2 - (eta_21 + eta_03) ** 2)

    phi_6 = (eta_20 - eta_02) * ((eta_30 + eta_12) ** 2 - (eta_21 + eta_03) ** 2) + 4 * eta_11 * (eta_30 + eta_12) * \
            (eta_21 + eta_03)

    phi_7 = (3 * eta_21 - eta_03) * (eta_30 + eta_12) * ((eta_30 + eta_12) ** 2 - 3 * (eta_21 + eta_03) ** 2) + \
            (3 * eta_12 - eta_03) * (eta_21 + eta_03) * (3 * (eta_30 + eta_12) ** 2 - (eta_21 + eta_03) ** 2)

    return (np.sign(phi_1) * np.log10(np.abs(phi_1)), np.sign(phi_2) * np.log10(np.abs(phi_2)),
            np.sign(phi_3) * np.log10(np.abs(phi_3)), np.sign(phi_4) * np.log10(np.abs(phi_4)),
            np.sign(phi_5) * np.log10(np.abs(phi_5)), np.sign(phi_6) * np.log10(np.abs(phi_6)),
            np.sign(phi_7) * np.log10(np.abs(phi_7)))


def compute_m_pq(input_img, p, q):
    m_pq = 0.0
    input_h, input_w = input_img.shape
    for x in range(input_h):
        for y in range(input_w):
            m_pq += x ** p * y ** q * input_img[x][y]

    return m_pq


def compute_mu_pq(input_img, p, q, x_bar, y_bar):
    mu_pg = 0.0
    input_h, input_w = input_img.shape
    for x in range(input_h):
        for y in range(input_w):
            mu_pg += (x - x_bar) ** p * (y - y_bar) ** q * input_img[x][y]

    return mu_pg


def compute_eta_pq(input_img, p, q, x_bar, y_bar, mu_00):
    mu_pq = compute_mu_pq(input_img, p, q, x_bar, y_bar)
    eta_pg = mu_pq / mu_00 ** ((p + q) / 2 + 1)

    return eta_pg


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
    (img_M1_phi_1, img_M1_phi_2, img_M1_phi_3, img_M1_phi_4, img_M1_phi_5, img_M1_phi_6,
     img_M1_phi_7) = invariant_moment(img_M1)
    (img_M2_phi_1, img_M2_phi_2, img_M2_phi_3, img_M2_phi_4, img_M2_phi_5, img_M2_phi_6,
     img_M2_phi_7) = invariant_moment(img_M2)
    (img_M3_phi_1, img_M3_phi_2, img_M3_phi_3, img_M3_phi_4, img_M3_phi_5, img_M3_phi_6,
     img_M3_phi_7) = invariant_moment(img_M3)
    (img_M4_phi_1, img_M4_phi_2, img_M4_phi_3, img_M4_phi_4, img_M4_phi_5, img_M4_phi_6,
     img_M4_phi_7) = invariant_moment(img_M4)
    (img_M5_phi_1, img_M5_phi_2, img_M5_phi_3, img_M5_phi_4, img_M5_phi_5, img_M5_phi_6,
     img_M5_phi_7) = invariant_moment(img_M5)
    (img_M6_phi_1, img_M6_phi_2, img_M6_phi_3, img_M6_phi_4, img_M6_phi_5, img_M6_phi_6,
     img_M6_phi_7) = invariant_moment(img_M6)
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
    print("phi_1, \tM1: {:.10f}, \tM2: {:.10f}, \tM3: {:.10f}, \tM4: {:.10f}, \tM5: {:.10f}, \tM6: {:.10f}".format(
        img_M1_phi_1, img_M2_phi_1, img_M3_phi_1, img_M4_phi_1, img_M5_phi_1, img_M6_phi_1
    ))
    print("phi_2, \tM1: {:.10f}, \tM2: {:.10f}, \tM3: {:.10f}, \tM4: {:.10f}, \tM5: {:.10f}, \tM6: {:.10f}".format(
        img_M1_phi_2, img_M2_phi_2, img_M3_phi_2, img_M4_phi_2, img_M5_phi_2, img_M6_phi_2
    ))
    print("phi_3, \tM1: {:.10f}, \tM2: {:.10f}, \tM3: {:.10f}, \tM4: {:.10f}, \tM5: {:.10f}, \tM6: {:.10f}".format(
        img_M1_phi_3, img_M2_phi_3, img_M3_phi_3, img_M4_phi_3, img_M5_phi_3, img_M6_phi_3
    ))
    print("phi_4, \tM1: {:.10f}, \tM2: {:.10f}, \tM3: {:.10f}, \tM4: {:.10f}, \tM5: {:.10f}, \tM6: {:.10f}".format(
        img_M1_phi_4, img_M2_phi_4, img_M3_phi_4, img_M4_phi_4, img_M5_phi_4, img_M6_phi_4
    ))
    print("phi_5, \tM1: {:.10f}, \tM2: {:.10f}, \tM3: {:.10f}, \tM4: {:.10f}, \tM5: {:.10f}, \tM6: {:.10f}".format(
        img_M1_phi_5, img_M2_phi_5, img_M3_phi_5, img_M4_phi_5, img_M5_phi_5, img_M6_phi_5
    ))
    print("phi_6, \tM1: {:.10f}, \tM2: {:.10f}, \tM3: {:.10f}, \tM4: {:.10f}, \tM5: {:.10f}, \tM6: {:.10f}".format(
        img_M1_phi_6, img_M2_phi_6, img_M3_phi_6, img_M4_phi_6, img_M5_phi_6, img_M6_phi_6
    ))
    print("phi_7, \tM1: {:.10f}, \tM2: {:.10f}, \tM3: {:.10f}, \tM4: {:.10f}, \tM5: {:.10f}, \tM6: {:.10f}".format(
        img_M1_phi_7, img_M2_phi_7, img_M3_phi_7, img_M4_phi_7, img_M5_phi_7, img_M6_phi_7
    ))
