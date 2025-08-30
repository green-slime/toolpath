import opencamlib as ocl  # OpenCAMLib主库
import time
from opencamlib import camvtk  # 可视化组件
import trimesh  # 用于加载OBJ文件
from tqdm import tqdm  # 用于显示进度条
import config as cfg
import os
import multiprocessing

def load_obj_as_stl(obj_filename):
    # 使用trimesh加载OBJ文件
    mesh = trimesh.load(obj_filename)
    # 导出为临时STL文件
    os.makedirs("temp", exist_ok=True)  # 确保临时目录存在
    stl_filename = "temp/temp.stl"
    mesh.export(stl_filename)
    # 使用OpenCAMLib加载STL
    stl = camvtk.STLSurf(stl_filename)
    polydata = stl.src.GetOutput()
    surface = ocl.STLSurf()
    camvtk.vtkPolyData2OCLSTL(polydata, surface)
    return surface

# 然后使用load_obj_as_stl替代前面示例中的load_surface_from_file函数

# 1. 加载STL文件（OpenCAMLib主要使用STL格式，但可以先将OBJ转为STL）
def load_surface_from_file(filename):
    stl = camvtk.STLSurf(filename)  # 加载STL文件
    polydata = stl.src.GetOutput()
    surface = ocl.STLSurf()
    camvtk.vtkPolyData2OCLSTL(polydata, surface)
    return surface

# 2. 定义刀具
def setup_cutter(cutter_diameter, length):
    # 创建球头刀具，您也可以选择其他类型如平底刀、锥形刀等
    cutter = ocl.BallCutter(cutter_diameter, length)
    return cutter

# 3. 生成平行路径（例如沿Y方向的）
def generate_parallel_paths(xmin, xmax, ymin, ymax, step_over):
    paths = []
    serialized_paths = []  # 用于存储序列化的路径
    Ny = int((ymax-ymin)/step_over) + 1
    dy = float(ymax-ymin)/(Ny-1)
    
    for n in range(0, Ny):
        path = ocl.Path()
        y = ymin + n*dy
        p1 = ocl.Point(xmin, y, 0)  # 线的起点
        p2 = ocl.Point(xmax, y, 0)  # 线的终点
        l = ocl.Line(p1, p2)
        path.append(l)
        paths.append(path)
        serialized_paths.append([xmin, xmax, y])
    
    return paths, serialized_paths

# 4. 执行自适应路径下切算法
def calculate_toolpath_apdc(surface, cutter, paths, sampling_distance=0.004, min_sampling_distance=0.001):
    apdc = ocl.AdaptivePathDropCutter()
    apdc.setSTL(surface)
    apdc.setCutter(cutter)
    apdc.setSampling(sampling_distance)      # 最大采样距离
    apdc.setMinSampling(min_sampling_distance)   # 最小采样距离
    
    cl_paths = []
    n_points = 0
    
    for path in tqdm(paths):
        apdc.setPath(path)
        apdc.run()
        cl_points = apdc.getCLPoints()
        n_points += len(cl_points)
        cl_paths.append(cl_points)
    
    return (cl_paths, n_points)

# 或使用 batch 计算
def calculate_toolpath_bdc(surface, cutter, paths, sample_num=300):
    
    cl_paths = []
    n_points = 0
    
    for path in tqdm(paths):
        # 为每个路径创建新的 BatchDropCutter 实例
        bdc = ocl.BatchDropCutter()
        bdc.setSTL(surface)
        bdc.setCutter(cutter)
        # 手动对路径进行采样
        sampled_points = sample_path_uniform(path, sample_num)
        
        # 添加采样点到 BatchDropCutter
        for point in sampled_points:
            bdc.appendPoint(point)
        
        # 运行 drop cutter 计算
        bdc.run()
        
        # 获取结果
        cl_points = bdc.getCLPoints()
        n_points += len(cl_points)
        cl_paths.append(cl_points)
    
    return (cl_paths, n_points)

def calculate_toolpath_bdc_combined(surface, cutter, serialized_paths, sample_num=300):
    """
    将所有路径合并后使用单个BDC实例进行计算
    
    Args:
        surface: STL表面
        cutter: 刀具
        serialized_paths: 序列化的路径列表 [[xmin, xmax, y], ...]
        sample_num: 每条路径的采样点数
        
    Returns:
        (cl_paths, n_points): 分离的路径列表和总点数
    """
    bdc = ocl.BatchDropCutter()
    bdc.setSTL(surface)
    bdc.setCutter(cutter)
    
    # 用于记录每条路径的起始索引
    path_indices = []
    current_index = 0
    
    # 为所有路径生成采样点并添加到BDC
    for path in tqdm(serialized_paths, desc="Adding points to BDC"):
        xmin, xmax, y = path
        dx = (xmax - xmin) / (sample_num - 1)
        
        # 记录当前路径的起始索引
        path_indices.append(current_index)
        
        # 添加当前路径的所有采样点
        for i in range(sample_num):
            x = xmin + i * dx
            bdc.appendPoint(ocl.CLPoint(x, y, -0.1))
            current_index += 1
            #print(f"Total points added to BDC: {current_index}")
    
    # 运行BDC计算
    print("Running BDC calculation...")
    bdc.run()
    
    # 获取所有计算结果
    all_cl_points = bdc.getCLPoints()
    print(f"BDC calculation completed, got {len(all_cl_points)} points")
    
    # 将结果分离回各条路径
    cl_paths = []
    for i, start_idx in enumerate(path_indices):
        if i < len(path_indices) - 1:
            # 不是最后一条路径
            end_idx = path_indices[i + 1]
            path_points = all_cl_points[start_idx:end_idx]
        else:
            # 最后一条路径
            path_points = all_cl_points[start_idx:]
        
        cl_paths.append(path_points)
    
    return (cl_paths, len(all_cl_points))
            

def sample_path_uniform(path, sample_num):
    """
    对路径进行均匀采样
    
    Args:
        path: [xmin, xmax, y]
        sampling_distance: 采样距离
        
    Returns:
        list: 采样点列表
    """
    xmin, xmax, y = path
    sampled_points = []
    dx = (xmax - xmin) / (sample_num - 1)  # 计算每个采样点之间的距离
    for i in range(sample_num):
        x = xmin + i * dx
        sampled_points.append(ocl.CLPoint(x, y, 0))
    
    return sampled_points

# 5. 过滤路径点以简化输出
def filter_toolpaths(cl_paths, tolerance=1e-5):
    cl_filtered_paths = []
    n_filtered = 0
    
    for cl_path in cl_paths:
        f = ocl.LineCLFilter()
        f.setTolerance(tolerance)
        for p in cl_path:
            p2 = ocl.CLPoint(p.x, p.y, p.z)
            f.addCLPoint(p2)
        f.run()
        filtered = f.getCLPoints()
        n_filtered += len(filtered)
        cl_filtered_paths.append(filtered)
    
    return (cl_filtered_paths, n_filtered)

def convert_paths_to_nested_list(cl_paths):
    """
    将OpenCAMLib路径点转换为嵌套列表结构
    
    参数:
    cl_paths -- 计算得到的CLPoint路径点列表
    
    返回:
    嵌套列表结构 [[p11,p12,...,p1n],[p21,...]]，每个点为[x,y,z]坐标
    """
    result = []
    
    for path in cl_paths:
        path_points = []
        for point in path:
            # 将每个CLPoint转换为[x,y,z]列表
            path_points.append([point.x, point.y, point.z])
        result.append(path_points)
    
    return result

def save_nested_list(nested_list, filename):
    """
    将嵌套列表结构保存到文件
    
    参数:
    nested_list -- 嵌套列表结构
    filename -- 保存的文件名
    """
    import pickle
    
    with open(filename, 'wb') as f:
        pickle.dump(nested_list, f)
    
    print(f"已将嵌套列表结构保存至 {filename}")

def load_nested_list(filename):

    import pickle
    
    with open(filename, 'rb') as f:
        nested_list = pickle.load(f)
    
    return nested_list

def save_toolpaths_to_file(cl_paths, filename):
    list = convert_paths_to_nested_list(cl_paths)
    os.makedirs(os.path.dirname(filename), exist_ok=True)  # 确保目录存在
    save_nested_list(list, filename)
    #print(list)

# 主流程
def main(need_mesh=False, device='cpu', foldername="toolpaths/"):
    if(need_mesh):
        from get_init_mesh import get_init_mesh
        # 生成初始网格
        get_init_mesh(sample_num=500, device=device, filename=cfg.init_mesh_filename)
    surface = load_obj_as_stl(cfg.init_mesh_filename)
    
    # 设置刀具参数（直径、刀具长度）
    cutter = setup_cutter(cfg.R*2, 3.0)
    
    # 确定加工范围
    xmin, xmax = 0, 1.0
    ymin, ymax = 0, 1.0
    step_over = 1.0 / cfg.ocl_path_num  # 根据配置文件中的路径数量计算步距
    
    # 生成路径
    paths, serialized_paths = generate_parallel_paths(xmin, xmax, ymin, ymax, step_over)
    
    # 计算刀具路径
    cl_paths, n_points = calculate_toolpath_bdc_combined(surface, cutter, serialized_paths, sample_num=cfg.sample_size)
    #cl_paths, n_points = calculate_toolpath_apdc(surface, cutter, paths, sampling_distance=0.005, min_sampling_distance=0.0001)
    #cl_paths = parallel_calculate_toolpath('temp/temp.stl', "ball", cfg.R*2, serialized_paths, sampling=0.001, min_sampling=0.0002)
    print(f"Generated {n_points} toolpath points")
    save_toolpaths_to_file(cl_paths, f"toolpaths/toolpaths_{cfg.ocl_path_num}_{cfg.R}_bdc_{cfg.surface_version}.pkl")
    
    # 过滤路径点
    #filtered_paths, n_filtered = filter_toolpaths(cl_paths)
    #print(f"After filtering: {n_filtered} toolpath points")
    #save_toolpaths_to_file(filtered_paths, f"toolpaths/filtered_toolpaths_{cfg.ocl_path_num}_{cfg.R}_.pkl")
    
    # 这里可以添加代码将路径输出为G代码或其他格式
    
if __name__ == "__main__":
    main()