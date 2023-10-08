import imageio.v3 as iio
import open3d as o3d

# 读取深度图片
depth_image = iio.imread('./image/ori/depth.png')
rgb_image = iio.imread('./image/ori/rgb.jpg')
# 设置相机参数（焦距、光心）
FX_DEPTH = 200
FY_DEPTH = 200
CX_DEPTH = 320
CY_DEPTH = 240

pcd = []
colors = []
height, width = depth_image.shape
for i in range(height):
    for j in range(width):
        z = depth_image[i][j]
        x = (j - CX_DEPTH) * z / FX_DEPTH
        y = (i - CY_DEPTH) * z / FY_DEPTH
        pcd.append([x, y, z])
        colors.append(rgb_image[i][j] / 255)

# 创造点云
pcd_o3d = o3d.geometry.PointCloud()
# 设置坐标和颜色
pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
pcd_o3d.colors = o3d.utility.Vector3dVector(colors)
# 可视化
o3d.visualization.draw_geometries([pcd_o3d])
