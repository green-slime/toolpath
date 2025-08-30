import torch
import torchvision.transforms as T
import config as cfg
from PIL import Image
from pathlib import Path
from scipy.spatial import Delaunay
import render_extension
import intersect_extension
import os
import numpy as np

# 图像大小和路径
img_size = cfg.img_size
img_path = cfg.img_path

transform2 = T.Compose([
    T.Resize((img_size,img_size)),
    T.CenterCrop((img_size, img_size)),
    T.ToTensor()
])

def prepare(device='cpu'):
    img = Image.open(img_path).convert('L')
    real_picture=transform2(img).squeeze().float().detach().to(device)
    real_sobel_result = render_extension.real_sobel(
        real_picture,
        img_size
    )
    return real_picture, real_sobel_result

def prepare_target_img(target_pos, info_list, device='cpu'):
    """
    准备目标图像，因为我们想要拟合的并非是真实图像，而是优化后的曲面渲染生成的图像。
    target_pos: [N², 2] 的张量，表示接收平面上的点位置
    device: 设备类型
    """
    render_result = get_render_result(target_pos, info_list)
    real_sobel_result = render_extension.real_sobel(
        render_result,
        img_size
    )
    return render_result, real_sobel_result


def triangle_area_tensor(xy_coords, triangles):
    x1, y1 = xy_coords[triangles[:, 0].long(), 0], xy_coords[triangles[:, 0].long(), 1]
    x2, y2 = xy_coords[triangles[:, 1].long(), 0], xy_coords[triangles[:, 1].long(), 1]
    x3, y3 = xy_coords[triangles[:, 2].long(), 0], xy_coords[triangles[:, 2].long(), 1]
    area = 0.5 * torch.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    return area

def get_neeeded_info(device, nu=50, nv=50, real_picture=None):
    """
    在训练循环前，获取需要的信息
    """
    if real_picture is None:
        return
    nu = nv = cfg.sample_size + 1  # 采样点数量
    with torch.no_grad():
        # 生成采样网格
        u = torch.linspace(0, 1, nu)
        v = torch.linspace(0, 1, nv)
        u_grid, v_grid = torch.meshgrid(u, v, indexing='ij')
        xy_data = torch.tensor(
            np.stack([u_grid.flatten(), v_grid.flatten()], axis=1),
            dtype=torch.float32,
            device=device
        )
        
        if not os.path.exists(cfg.temp_triangles_filename):
            # Delaunay三角化
            xy_np = xy_data.detach().cpu().numpy()
            triangulation = Delaunay(xy_np)
            triangles = triangulation.simplices
            triangles_tensor = torch.tensor(triangles, dtype=torch.int).to(device)
            # 保存三角形数据
            os.makedirs(os.path.dirname(cfg.temp_triangles_filename), exist_ok=True)
            torch.save(triangles_tensor, cfg.temp_triangles_filename)
        else:
            triangles_tensor = torch.load(cfg.temp_triangles_filename).to(device)  # 从文件加载三角形数据
            
        # 计算三角形面积
        tri_area = triangle_area_tensor(xy_data, triangles_tensor).to(device)
        #print(triangles_tensor.shape, proj_xy.shape)
        # 准备渲染参数
        #total_gray_value = torch.tensor(1500000., dtype=torch.float32).cuda()
        glass_plane_edge_length = torch.tensor([1.0], dtype=torch.float32,device=device) # 1*1 的透镜
        target_plane_edge_length = torch.tensor([1.0], dtype=torch.float32,device=device) # 1*1 的接收平面
        gray_value_per_area = (real_picture.sum()/glass_plane_edge_length**2).to(device)
    # print(f"Detached? {xy_data.requires_grad, tri_area.requires_grad, triangles_tensor.requires_grad, total_gray_value.requires_grad, real_picture.requires_grad}")
    return [xy_data, tri_area, triangles_tensor, gray_value_per_area, target_plane_edge_length, real_picture]

def modify_infolist(info_list, target_picture):
    """
    当将 info_list 中的 real_picture 替换为 target_picture 时，调用此函数。
    更改 info_list 中的 gray_value_per_area 和 real_picture。
    """
    """
    仔细想想，这里好像不需要 modify 光通量。
    理想状态中的光通量经过优化后的曲面可能变少（射出目标范围），
    我们应该是光通量依旧采用原图像的，但是 render loss 的计算
    采用目标图像的对应像素以及 Sobel 结果。
    """
    #gray_value_per_area = info_list[3] * target_picture.sum() / info_list[5].sum()
    info_list[5] = target_picture
    #info_list[3] = gray_value_per_area


def get_render_result(proj_xy, info_list) -> torch.Tensor:
    # 返回：render_result: Tensor:(img_size, img_size), 表示渲染的图像
    render_result = render_extension.render(
        info_list[1],
        info_list[2],
        proj_xy.clone().detach(),
        info_list[3],
        info_list[4],
        img_size
    )[0]
    return render_result
import time
def get_diff(proj_xy, info_list, sobel_result):
    # 返回：points_diff: (n, 2), 表示 loss 对 proj_xy 的梯度
    # render_diff: (img_size, img_size), 表示 loss 
    diff_result = render_extension.diff(
        info_list[1],
        info_list[2],
        proj_xy.clone().detach(),
        info_list[3],
        info_list[5],
        sobel_result[0],
        sobel_result[1],
        info_list[4],
        img_size
    ) # output: (render_diff,points_diff,tri_area_diff,render_result,points_sobel_diff, render_sobel_x, render_sobel_y, points_diff_l1)

    sobel_scale = cfg.sobel_scale
    l1_scale = cfg.l1_scale
    points_diff = (diff_result[1] + sobel_scale * diff_result[4] + l1_scale * diff_result[7]).detach().requires_grad_(False)
    render_diff = torch.sum(diff_result[0]**2)
    sobel_diff = torch.sum(diff_result[5]**2) + torch.sum(diff_result[6]**2)
    render_diff_l1 = torch.sum(torch.abs(diff_result[0]))
    render_diff = render_diff + sobel_diff * sobel_scale + render_diff_l1 * l1_scale
    return points_diff, render_diff

def get_diff_2(proj_xy, info_list, sobel_result):
    diff_result2 = render_extension.new_diff(
        info_list[1],
        info_list[2],
        proj_xy.clone().detach(),
        info_list[3],
        info_list[5],
        info_list[4],
        img_size
    ) # output: (render_result, render_diff, proj_xy_grad, tri_area_grad)
    return diff_result2[2].detach().requires_grad_(False), diff_result2[1]
   


def prepare_for_render(device='cpu'):
    real_picture, real_sobel_result = prepare(device=device)
    info_list = get_neeeded_info(device=device, real_picture=real_picture)
    proj_xy = info_list[0].clone().detach().requires_grad_(True)
    points_diff = get_diff(proj_xy, info_list, real_sobel_result)
    #intersect_extension.intersect_with_grad_for_render(Ps, path_points, grad_render_to_yj, R, MAX_HEIGHT, n1, float n2, float z_plane_height)