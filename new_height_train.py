import new_intersect
import bvh
import render
import torch
import utils
import config as cfg
from init import init
from init_with_camlib import init_with_camlib, init_sample_points
import utils_draw
import get_target_pos
from torch import optim
import prepare_for_render
import evaluate_surface_shape

def get_fixed_receiver_points(device='cpu'):
    """
    获取固定的接收器点，形状 (N², 2)，
    输出张量的 requires_grad 属性为 False。
    """
    return utils.make_2d_sample_points(cfg.sample_size, device=device).requires_grad_(False)

def get_receiver_points(Ps, path_points):
    heights, normals, _ = new_intersect.intersect(Ps, path_points, cfg.R, cfg.zmax)
    Ps_xy = Ps[:, :, :2].reshape(-1, 2)  # 取出采样点的 x 和 y 分量
    surface_points = torch.cat([Ps_xy, heights.unsqueeze(-1)], dim=-1)  # 形状 (N, 3)，N 是采样点的总数
    receiver_points = render.trace_rays_through_surface(surface_points, normals, z_receiver=cfg.z_of_receiver, n1=cfg.n1, n2=cfg.n2)
    return receiver_points

def get_dirs(Ps, path_points):
    heights, normals, dirs = new_intersect.intersect(Ps, path_points, cfg.R, cfg.zmax)
    Ps_xy = Ps[:, :, :2].reshape(-1, 2)  # 取出采样点的 x 和 y 分量
    surface_points = torch.cat([Ps_xy, heights.unsqueeze(-1)], dim=-1)  # 形状 (N, 3)，N 是采样点的总数
    
    return surface_points, dirs

def height_loss_new(Ps, path_points, target_height):
    """
    计算高度损失，使用新的 CUDA 扩展。
    """
    _, _, loss, grad = new_intersect.intersect_with_height_grad(Ps, cfg.R, path_points, MAX_HEIGHT=cfg.zmax, target_height=target_height)
    return loss, grad

def use_pt(pt_name, device='cpu'):
    """
    使用预训练的路径点，如果不存在则返回 None。
    """
    try:
        path_points = torch.load(pt_name, map_location=device)
        print(f"使用预训练的路径点: {pt_name}")
        return path_points.clone().detach().requires_grad_(True)  # 返回一个叶子节点
    except FileNotFoundError:
        print(f"未找到预训练的路径点: {pt_name}")
        exit()
        
def make_obj(sample_size, path_points, device='cpu', output_file=cfg.obj_filename):
    from init import init_sample_points
    Ps = init_sample_points(sample_size, device=device)  # 初始化采样点
    heights, normals, _ = new_intersect.intersect(Ps, path_points, cfg.R, cfg.zmax)
    #heights, normals = batch_intersect(Ps, cfg.R, sample_size, x_grid, path_points_yz, batch_size = cfg.batch_size, device=device, alpha=cfg.alpha, MAX_HEIGHT=cfg.zmax)
    #print(normals.reshape(-1, sample_size +1, 3)[:,8:12,:])
    #exit()
    #utils.create_closed_grid(sample_size, heights, normals, min_height=None, output_file=output_file)
    utils.save_to_obj(heights, normals, nu=sample_size + 1, nv=sample_size + 1, filename=output_file)  # 保存 OBJ 文件
    
import os
import new_surface_evaluator
def just_show_result(pretrained_pt=None):
    if pretrained_pt is None:
        print("请提供预训练的路径点文件名。")
        exit()
    foldername = os.path.dirname(pretrained_pt)
    with torch.no_grad():
        device = utils.cuda_init(cfg.cuda_num)
        Ps = init_sample_points(cfg.sample_size, device=device)
        init_path_points = torch.load(f"centers/centers_{cfg.ocl_path_num}_{cfg.R}_bdc_{cfg.surface_version}.pt", map_location=device)
        #Ps, init_path_points = init_with_camlib(device, use_x=True)  # 初始化采样点和路径点
        trained_path_points = use_pt(pretrained_pt, device=device)
        fixed_receiver_points, fixed_heights = get_target_pos.get_target_pos(device=device)  # 获取目标接收器点
        surface_points1, dirs = get_dirs(Ps, init_path_points)  # 获取当前方向向量
        utils_draw.visualize_paths_and_surface(surface_points1, init_path_points, grid_size=cfg.sample_size+1, filename=f"{foldername}/show_result/paths_init.png", path_sample_ratio=0.05, surface_alpha=0.3, path_colors=['red'])  # 绘制路径和表面
        surface_points2, dirs = get_dirs(Ps, trained_path_points)  # 获取当前方向向量
        utils_draw.visualize_paths_and_surface(surface_points2, trained_path_points, grid_size=cfg.sample_size+1, filename=f"{foldername}/show_result/paths_final.png", path_sample_ratio=0.05, surface_alpha=0.3, path_colors=['red'])  # 绘制路径和表面
        # 绘制复合图像
        evaluator = new_surface_evaluator.surface_shape_evaluator(device, Ps, fixed_heights)
        evaluator.set_init_heights(surface_points1[:, 2])
        evaluator.set_final_heights(surface_points2[:, 2])
        #evaluator.visualize_surfaces_simplified(save_path=f'{foldername}/surface_comparison.png')
        # 调用新函数
        evaluator.visualize_mae_distribution(save_path=f'{foldername}/mae_distribution_analysis.png')

def train(pretrained_pt=None):
    device = utils.cuda_init(cfg.cuda_num)
    Ps, path_points = init_with_camlib(device, use_x=True)  # 初始化采样点和路径点
    if pretrained_pt is not None:
        path_points = use_pt(pretrained_pt, device=device)
    with torch.no_grad():
        fixed_receiver_points, fixed_heights = get_target_pos.get_target_pos(device=device)  # 获取目标接收器点
        
        surface_points, dirs = get_dirs(Ps, path_points)  # 获取当前方向向量
        
        #print(f"target height 第一行: {fixed_heights.reshape(cfg.sample_size+1, cfg.sample_size+1)[:,0]}")
        #print(f"path_points_yz 第一行：{path_points_yz[0,:,1]}")
        #print(f"surface points 第一行： {surface_points.reshape(cfg.sample_size+1, cfg.sample_size+1, 3)[:,0,2]}")
        utils_draw.visualize_paths_and_surface(surface_points, path_points, grid_size=cfg.sample_size+1, filename=f"{cfg.output_foldername}/surface_vis/paths_init.png", path_sample_ratio=0.02, surface_alpha=0.3)  # 绘制路径和表面
        surface_shape_evaluator = evaluate_surface_shape.surface_shape_evaluator(device, Ps, fixed_heights, surface_points[:, 2])  # 创建曲面形状评估器
        surface_shape_evaluator.print_evaluation_summary(save_to_file=f"{cfg.output_foldername}/surface_summary1.txt")  # 打印评估结果
        surface_shape_evaluator.visualize_surfaces_comprehensive(save_path=f'{cfg.output_foldername}/shape_evaluate/surface_comparison.png')
        utils_draw.visualize_surface_vectors_3d(surface_points, dirs, grid_size=cfg.sample_size+1, filename=f"{cfg.output_foldername}/surface_vis/surface_vectors_-1.png")  # 绘制表面向量

        receiver_points = get_receiver_points(Ps, path_points)  # 获取当前接收器点
        
    optimizer = torch.optim.Adam([path_points], cfg.learning_rate*0.1, betas=(0.9, 0.99), eps=1e-8, amsgrad=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=50, cooldown=0, verbose=True, eps=1e-12)
    
    losses = []
    change_at_epoch = -1  # 不切换

    utils.save_config_to_file(cfg.output_foldername, "config.txt")  # 保存配置文件
    utils.save_config_to_file(cfg.output_foldername, f"{cfg.extra_config_filename}.txt", readfilename=f"{cfg.extra_config_filename}.py")  # 保存多步运行配置文件
 
    for epoch in range(cfg.max_epochs):
        #print(f"path_points_yz: {path_points_yz}")
        optimizer.zero_grad()
        if epoch == change_at_epoch:
            optimizer.param_groups[0]['lr'] = cfg.learning_rate * 0.1  # 重设学习率
        if epoch > change_at_epoch:
            #height_loss, grad = height_loss_with_cuda(Ps, x_grid, path_points_yz, fixed_heights, device=device)
            height_loss, grad = height_loss_new(Ps, path_points, fixed_heights)
        else:
            pass
        #ot_loss, grad = ot_loss_with_cuda(Ps, x_grid, path_points_yz, fixed_receiver_points, device=device)
        if path_points.grad is None:
            # 第一次迭代需要初始化梯度张量
            path_points.grad = torch.zeros_like(path_points)
        path_points.grad.copy_(grad)

        #loss =  ot_loss + overlayloss * cfg.overlay_weights
        if epoch > change_at_epoch:
            loss = height_loss 
        else:
            pass
        #loss.backward()
        #print(f"Gradient: {path_points_yz.grad}")  # 打印梯度
        optimizer.step()
        losses.append(loss.item())
        scheduler.step(loss)  # 更新学习率
        if epoch % cfg.print_epochs == 0 or epoch == cfg.max_epochs - 1:
            #print(f"Epoch {epoch + 1}/{cfg.max_epochs}, Loss: {loss.item():.6f}")
            if epoch > change_at_epoch:
                print(f"Epoch {epoch + 1}/{cfg.max_epochs}, height Loss: {height_loss.item():.6f}")
            else:
                pass
            #print(f"Epoch {epoch + 1}/{cfg.max_epochs}, Loss: {loss.item():.6f}, OT Loss: {ot_loss.item():.6f}, Overlay Loss: {overlayloss.item():.6f}")
            #exit()
            current_lr = optimizer.param_groups[0]['lr']
            if(current_lr < 1e-12):
                print(f"Learning rate too small: {current_lr}, stopping training.")
                with torch.no_grad():
                
                    break
        #if epoch % cfg.draw_epochs == 0 or epoch == cfg.max_epochs - 1:
            #print(f"Epoch {epoch + 1}/{cfg.max_epochs}, Path Points YZ: {path_points_yz}")
        if (epoch >= 0 and epoch % cfg.draw_epochs == 0) or epoch == cfg.max_epochs - 1:
            with torch.no_grad():
                receiver_points = get_receiver_points(Ps, path_points)  # 获取当前接收器点
                surface_points, dirs = get_dirs(Ps, path_points)  # 获取当前方向向量
                utils_draw.visualize_surface_vectors_3d(surface_points, dirs, grid_size=cfg.sample_size+1, filename=f"{cfg.output_foldername}/surface_vis/surface_vectors_{epoch}.png")  # 绘制表面向量
                
    with torch.no_grad():
        # 绘制 loss 曲线
        utils_draw.drawLossCurve(losses, cfg.losses_filename)
        # 保存最终的路径点
        print(path_points)
        heights, normals, _ = new_intersect.intersect(Ps, path_points, cfg.R, cfg.zmax)
        print(f"heights:{heights}, normals:{normals}")
        torch.save(path_points, f"{cfg.output_foldername}/path_points.pt")  # 保存路径点
        utils_draw.visualize_paths_and_surface(surface_points, path_points, grid_size=cfg.sample_size+1, filename=f"{cfg.output_foldername}/surface_vis/paths_final.png", path_sample_ratio=0.02, surface_alpha=0.3)  # 绘制路径和表面
        surface_shape_evaluator.update_our_surface(heights)
        surface_shape_evaluator.print_evaluation_summary(save_to_file=f"{cfg.output_foldername}/surface_summary2.txt")  # 打印评估结果
        surface_shape_evaluator.visualize_surfaces_comprehensive(save_path=f'{cfg.output_foldername}/shape_evaluate2/surface_comparison.png')
        #make_obj(cfg.sample_size_for_obj, path_points_yz, x_grid, device=device, output_file=f"{cfg.output_foldername}/output.obj")  # 保存 OBJ 文件
        
def get_obj():
    device = utils.cuda_init(cfg.cuda_num)
    path_points_yz = torch.load(f"{cfg.output_foldername}/path_points_yz.pt")
    x_grid = bvh.make_x_grid(cfg.path_size, cfg.sample_size, cfg.R, device=device)  # 预计算可能相交的路径点
    make_obj(cfg.sample_size_for_obj, path_points_yz, x_grid, device=device, output_file=f"{cfg.output_foldername}/output2.obj")  # 
    
if __name__ == "__main__":
    just_show_result(pretrained_pt="./output/large_600_2000_0.01_large_23/"+"path_points.pt")  # 显示结果
