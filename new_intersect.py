import torch
from tqdm import tqdm
import bvh
import intersect_extension
import config as cfg
import render
import prepare_for_render
from time import time
import new_intersect_extension    

def intersect(Ps, path_points, R, MAX_HEIGHT=1.0):
    """
    求交函数，返回高度、法向量、损失和梯度。
    """
    height, normal, dirs = new_intersect_extension.intersect(Ps.contiguous(), path_points.contiguous(), R, MAX_HEIGHT)
    return height, normal, dirs

def intersect_with_height_grad(Ps, R, path_points, MAX_HEIGHT=1.0, target_height=None):
    """
    新的 cuda 代码下不需要 batch 处理，直接对所有采样点进行处理。
    """
    gouge_weight = 5.0
    #from time import time
    #start_time = time()
    if target_height is not None:
        # 生成 x 分量并拼接到 path_yzs 上
        
        height2, normal2, loss2, grad2 = new_intersect_extension.intersect_with_height_grad(Ps.contiguous(), path_points.contiguous(), R, MAX_HEIGHT, target_height.contiguous(), gouge_weight)
        loss = loss2.sum()
        #print(f"求交 time: {time() - start_time:.4f} s")
        #print(f"target height 第一行: {target_height.reshape(sample_size+1, sample_size+1)[:,0]}")
        #print(f"height2 第一行：{height2.reshape(sample_size+1, sample_size+1)[:,0]}")
        return height2, normal2, loss, grad2
    else:
        raise NotImplementedError("需要提供 target_height 进行高度监督。")

def intersect_with_render_grad(Ps, R, path_points, sobel_result, infolist, MAX_HEIGHT=1.0):
    """
    功能与下同，但由渲染 loss 监督。
    """

    heights, normals, dirs = intersect(Ps.contiguous(), path_points.contiguous(), R, MAX_HEIGHT)  # 求交
    with torch.no_grad():
        receiver_points = render.trace_rays_through_surface(torch.cat([Ps[:, :, :2].reshape(-1, 2), heights.unsqueeze(-1)], dim=-1), normals, cfg.z_of_receiver, n1=cfg.n1, n2=cfg.n2)  # 形状 ((sample_size + 1) * (sample_size + 1), 2)
        points_diff, render_diff = prepare_for_render.get_diff(receiver_points, infolist, sobel_result)  # 计算渲染损失
        points_diff = points_diff.reshape(-1, 2)  # 将差值展平为二维
        grad = new_intersect_extension.intersect_with_grad_for_render(Ps.contiguous(), path_points.contiguous(), points_diff.contiguous(), R, MAX_HEIGHT, cfg.n1, cfg.n2, cfg.z_of_receiver)[0]  # 计算梯度
        loss = torch.norm(render_diff, p=2)  # 累加损失
    return loss, grad, receiver_points