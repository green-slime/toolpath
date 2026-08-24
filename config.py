# gpu 设置
cuda_num = 3

# 曲面基本参数
zmin = -1.0
zmax = 0.3

z_init = zmax

# 各采样参数
path_size = 600  # 单条路径上采样点数
path_num = 600  # 路径数量
sample_size = 2000  # 网格采样点数
sample_size_for_obj = 1000 # 制作 obj 文件采样数

# 目标曲面描述配置字典
surface_configs = {
  3: {
    "img_name": "gray_circle",
    "img_path": "./img/gray_circle_200.png"
  },
  7: {
    "img_name": "ustc_400", 
    "img_path": "./img/ustc_400.png"
  }
}

# 当前使用的配置
surface_version = 3
img_name = surface_configs[surface_version]["img_name"]
img_path = surface_configs[surface_version]["img_path"]
control_points_name = f'control_points{surface_version}.pth'  # 控制点文件名
wij_name = f'wij{surface_version}.pth'  # wij 文件名
init_mesh_filename = f'./init_mesh/init_mesh{surface_version}.obj'  # 初始网格文件名

# 渲染参数
img_size = 200  # 渲染图像大小
#img_path = f'./img/zju.png'  # 目标图像位置
#img_path = f'./img/einstein_200.png'  # 目标图像位置
#img_path = f'./img/gray_circle_200.png'  # 目标图像位置
#img_path = f'./img/einstein_200_400.png'  # 目标图像位置
#img_path = f'./img/6_400.png'  # 目标图像位置
#img_path = f'./img/lena_200.png'  # 目标图像位置
#img_path = f'./img/ustc_400.png'  # 目标图像位置
#img_path = f'./img/lena_200_400.png'  # 目标图像位置
sobel_scale = 1.0  # Sobel算子缩放比例
l1_scale = 0.2

# 光滑性约束
reg_weight = 0.1  # 正则约束损失权重

# 路径样条曲线控制点密度
control_points_ratio = 2  # path_len // control_points_ratio = 控制点数量

# 算法参数
alpha = 1e7  # 求交算法中的 alpha 参数（for softmin），已不使用

# 刀头半径
R = 0.02

# 等残高法残高限制
h = 1e-3

# 光学参数
n1 = 1.5 # 介质折射率
n2 = 1.0 # 空气折射率
z_of_receiver = 3.0 # 接收平面 z 坐标

# 训练超参数和配置
learning_rate = 5e-6
ot_learning_rate = 1e-4  # OT loss 学习率
normal_learning_rate = 5e-5  # normal loss 学习率
change_at_epoch = -1  # 切换到渲染 loss 的 epoch
max_epochs = 2000  # 最大训练轮数
overlay_weights = 1.0 # overlay loss的权重
draw_epochs = 100  # 绘图间隔
print_epochs = 10   # 打印间隔
batch_size = 3000    # 每个 batch 取的采样点列数


# OCL 路径初始化设置
need_toolpath_preprocess = False  # 是否需要对路径进行预处理
need_mesh = True  # 是否需要生成网格
ocl_path_num = path_num  # OCL 路径数量

# 输出配置
output_foldername = f"output/large_{ocl_path_num}_{sample_size}_0.01_height_3/"  # 输出文件夹
temp_triangles_filename = f"temp/triangles_{sample_size}.pt"  # 临时三角形数据文件
losses_filename = f"{output_foldername}/losses.png"  # 损失记录文件
pairpos_foldername = f"{output_foldername}/pair_pos/" # 可视化投影结果
obj_filename = f"{output_foldername}/output.obj"  # 输出的 OBJ 文件

extra_config_filename = "test_direct_train"  # 额外配置文件名