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
import new_constraints

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

def get_render_loss(Ps, path_points, sobel_result, infolist):
    loss, grad, receiver_points = new_intersect.intersect_with_render_grad(Ps, cfg.R, path_points, sobel_result, infolist, MAX_HEIGHT=cfg.zmax)
    return loss, grad, receiver_points

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
    
def prepare_for_rendering(device, fixed_receiver_points):
    """
    仔细想想，这里好像不需要 modify 光通量。
    理想状态中的光通量经过优化后的曲面可能变少（射出目标范围），
    我们应该是光通量依旧采用原图像的，但是 render loss 的计算
    采用目标图像的对应像素以及 Sobel 结果。
    """
    print("准备渲染所需的信息...")
    real_picture, real_sobel_result = prepare_for_render.prepare(device=device)
    info_list = prepare_for_render.get_neeeded_info(device=device, real_picture=real_picture)
    target_picture, target_sobel_result = prepare_for_render.prepare_target_img(fixed_receiver_points, info_list, device=device)  # 准备目标图像和 Sobel 结果
    prepare_for_render.modify_infolist(info_list, target_picture)  # 替换 infolist 中的 real_picture 为 target_picture
    return info_list, real_picture, target_picture, real_sobel_result, target_sobel_result
    
import os
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
        
        """ surface_points1, dirs = get_dirs(Ps, trained_path_points)  # 获取当前方向向量
        utils_draw.visualize_paths_and_surface(surface_points1, trained_path_points, grid_size=cfg.sample_size+1, filename=f"{foldername}/surface_vis/paths_init.png", path_sample_ratio=0.05, surface_alpha=0.3, path_colors=['red'])  # 绘制路径和表面
        surface_points2, dirs = get_dirs(Ps, trained_path_points)  # 获取当前方向向量
        utils_draw.visualize_paths_and_surface(surface_points2, trained_path_points, grid_size=cfg.sample_size+1, filename=f"{foldername}/surface_vis/paths_trained.png", path_sample_ratio=0.05, surface_alpha=0.3, path_colors=['red'])  # 绘制路径和表面
        exit() """
        
        info_list, real_picture, target_picture, real_sobel_result, target_sobel_result = prepare_for_rendering(device, fixed_receiver_points)  # 准备渲染所需的信息
        # 准备绘制图像
        init_receiver_points = get_receiver_points(Ps, init_path_points)
        init_render_result = prepare_for_render.get_render_result(init_receiver_points, info_list)
        #utils_draw.drawThreeMap(target_picture, init_render_result, f'{foldername}/show_result/compare_init.png', 'Target', 'Render Result at init.')  # 绘制目标图像和渲染结果的对比图
        trained_receiver_points = get_receiver_points(Ps, trained_path_points)  # 获取当前接收器点
        trained_render_result = prepare_for_render.get_render_result(trained_receiver_points, info_list)
        #utils_draw.drawThreeMap(target_picture, trained_render_result, f'{foldername}/show_result/compare_trained.png', 'Target', 'Render Result after training.')
        # 准备输出指标
        utils_draw.drawFiveMap(target_picture, init_render_result, trained_render_result, f'{foldername}/show_result/compare.png')
        utils.calculate_img_metrics(target_picture, init_render_result, f"{foldername}/show_result/init_image_metrics.txt")  # 计算图像指标
        utils.calculate_img_metrics(target_picture, trained_render_result, f"{foldername}/show_result/trained_image_metrics.txt")  # 计算图像指标
        

def train(pretrained_pt=None):
    device = utils.cuda_init(cfg.cuda_num)
    Ps, path_points = init_with_camlib(device, use_x=True)  # 初始化采样点和路径点
    if pretrained_pt is not None:
        path_points = use_pt(pretrained_pt, device=device)
    
    with torch.no_grad():
        fixed_receiver_points, fixed_heights = get_target_pos.get_target_pos(device=device)  # 获取目标接收器点
        #fixed_receiver_points = mocked_fixed_target(device)  # 使用模拟的固定接收器点
        surface_points, dirs = get_dirs(Ps, path_points)  # 获取当前方向向量
        
        utils_draw.visualize_paths_and_surface(surface_points, path_points, grid_size=cfg.sample_size+1, filename=f"{cfg.output_foldername}/surface_vis/paths_init.png", path_sample_ratio=0.05, surface_alpha=0.3, path_colors=['red'])  # 绘制路径和表面

        info_list, real_picture, target_picture, real_sobel_result, target_sobel_result = prepare_for_rendering(device, fixed_receiver_points)  # 准备渲染所需的信息
        
        utils_draw.drawHeatMap(real_picture, f"{cfg.output_foldername}/real_picture.png", title="Real Picture")  # 绘制真实图像
        utils_draw.drawHeatMap(target_picture, f"{cfg.output_foldername}/target_picture.png", title="Target Picture")  # 绘制真实图像

        receiver_points = get_receiver_points(Ps, path_points)
        render_result = prepare_for_render.get_render_result(receiver_points, info_list)
        utils_draw.drawTwoHeatMap(target_picture, render_result, path=f'{cfg.output_foldername}/render_result/compare_-1.png', title1='Target', title2=f'Render Result at init.')

    if cfg.change_at_epoch < 0: # 不进行 ot 的前提下
        start_learning_rate = cfg.learning_rate
    else:
        start_learning_rate = cfg.ot_learning_rate
    optimizer = torch.optim.Adam([path_points], start_learning_rate, betas=(0.8, 0.95), eps=1e-8, amsgrad=False)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=50, cooldown=0, verbose=True, eps=1e-9)
    # optimizer = torch.optim.SGD([path_points_yz], lr=1e-6, momentum=0, weight_decay=1e-3)
    """ optimizer = torch.optim.AdamW(
        [path_points_yz],  
        cfg.learning_rate,  
        betas=(0.8, 0.999),  # 动量参数
        eps=1e-8,            # 数值稳定性
        weight_decay=1e-3,    # 直接作为权重衰减系数 
        amsgrad=True
    ) """
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=100, cooldown=0, verbose=True, eps=1e-9)
    losses = []
    change_at_epoch = -1  # 不切换
    #change_at_epoch = cfg.change_at_epoch

    # 保存配置文件
    utils.save_config_to_file(cfg.output_foldername, "config.txt")  # 保存配置文件
    utils.save_config_to_file(cfg.output_foldername, f"{cfg.extra_config_filename}.txt", readfilename=f"{cfg.extra_config_filename}.py")  # 保存多步运行配置文件
 
    for epoch in range(cfg.max_epochs):
        #print(f"path_points_yz: {path_points_yz}")
        optimizer.zero_grad()
        if epoch == change_at_epoch:
            optimizer.param_groups[0]['lr'] = cfg.learning_rate   # 重设学习率
            print("Switching to render loss optimization...")
        if epoch < change_at_epoch:
            pass
        else:
            render_loss, grad, receiver_points = get_render_loss(Ps, path_points, target_sobel_result, info_list)
        #ot_loss, grad = ot_loss_with_cuda(Ps, x_grid, path_points_yz, fixed_receiver_points, device=device)
        #regularization_loss = new_constraints.compute_curvature_loss(path_points, cfg.reg_weight)  # 距离正则化损失
        regularization_loss = new_constraints.compute_laplacian_smoothness_loss(path_points, laplacian_weight=cfg.reg_weight)  # 拉普拉斯正则化损失
        regularization_loss.backward()
        #angle_loss = new_constraints.compute_angle_smoothness_loss_vectorized(path_points, angle_weight=cfg.angle_weight)  # 角度变化约束损失
        #angle_loss.backward()
        if path_points.grad is None:
            path_points.grad = grad.clone()
        else:
            if(epoch % 100 == 0):
                with torch.no_grad():
                    print(f"render grad max: {grad.abs().max()}, min: {grad.abs().min()}, norm: {grad.norm()}")
                    print(f"reg grad max: {path_points.grad.abs().max()}, min: {path_points.grad.abs().min()}, norm: {path_points.grad.norm()}")
            path_points.grad += grad  
        #loss = ot_loss
        #loss =  ot_loss + overlayloss * cfg.overlay_weights
        if epoch < change_at_epoch:
            pass
        else:
            loss = render_loss + regularization_loss   # 使用渲染损失和覆盖损失的加权和
        #loss.backward()
        #print(f"Gradient: {path_points_yz.grad}")  # 打印梯度
        optimizer.step()
        losses.append(loss.item())
        scheduler.step(loss)  # 更新学习率
        if epoch % cfg.print_epochs == 0 or epoch == cfg.max_epochs - 1:
            #print(f"Epoch {epoch + 1}/{cfg.max_epochs}, Loss: {loss.item():.6f}")
            if epoch < change_at_epoch:
                pass
            else:
                print(f"Epoch {epoch + 1}/{cfg.max_epochs}, render Loss: {render_loss.item():.6f}, regularization Loss: {regularization_loss.item():.6f}, Total Loss: {loss.item():.6f}")
            #print(f"Epoch {epoch + 1}/{cfg.max_epochs}, Loss: {loss.item():.6f}, OT Loss: {ot_loss.item():.6f}, Overlay Loss: {overlayloss.item():.6f}")
            #exit()
            current_lr = optimizer.param_groups[0]['lr']
            if(current_lr < 1e-9):
                print(f"Learning rate too small: {current_lr}, stopping training.")
                with torch.no_grad():
                    receiver_points = get_receiver_points(Ps, path_points)  # 获取当前接收器点
                    render_result = prepare_for_render.get_render_result(receiver_points, info_list)
                    #utils_draw.drawHeatMap(render_result, f"{cfg.output_foldername}/render_result/render_result_{epoch}.png", title=f"Render Result at Epoch {epoch}")  
                    utils_draw.drawTwoHeatMap(target_picture, render_result, path=f'{cfg.output_foldername}/render_result/compare_{epoch}.png', title1='Target', title2=f'Render Result at epoch {epoch}')
                    break
        #if epoch % cfg.draw_epochs == 0 or epoch == cfg.max_epochs - 1:
            #print(f"Epoch {epoch + 1}/{cfg.max_epochs}, Path Points YZ: {path_points_yz}")
        if (epoch >= 0 and epoch % cfg.draw_epochs == 0) or epoch == cfg.max_epochs - 1:
            with torch.no_grad():
                receiver_points = get_receiver_points(Ps, path_points)  # 获取当前接收器点
                render_result = prepare_for_render.get_render_result(receiver_points, info_list)
                surface_points, dirs = get_dirs(Ps, path_points)  # 获取当前方向向量
                utils_draw.visualize_surface_vectors_3d(surface_points, dirs, grid_size=cfg.sample_size+1, filename=f"{cfg.output_foldername}/surface_vis/surface_vectors_{epoch}.png")  # 绘制表面向量
                #utils_draw.drawHeatMap(render_result, f"{cfg.output_foldername}/render_result/render_result_{epoch}.png", title=f"Render Result at Epoch {epoch}")
                utils_draw.drawTwoHeatMap(target_picture, render_result, path=f'{cfg.output_foldername}/render_result/compare_{epoch}.png', title1='Target', title2=f'Render Result at epoch {epoch}')  # 绘制目标图像和渲染结果的对比图
    with torch.no_grad():
        # 绘制 loss 曲线
        utils_draw.drawLossCurve(losses, cfg.losses_filename)
        # 保存最终的路径点
        print(path_points)
        heights, normals, _ = new_intersect.intersect(Ps, path_points, cfg.R, cfg.zmax)
        print(f"heights:{heights}, normals:{normals}")
        torch.save(path_points, f"{cfg.output_foldername}/path_points.pt")  # 保存路径点
        surface_points, dirs = get_dirs(Ps, path_points)  # 获取当前方向向量
        utils_draw.visualize_paths_and_surface(surface_points, path_points, grid_size=cfg.sample_size+1, filename=f"{cfg.output_foldername}/surface_vis/paths_final.png", path_sample_ratio=0.05, surface_alpha=0.3, path_colors=['red'])   # 绘制路径和表面
        just_show_result(pretrained_pt=f"{cfg.output_foldername}/path_points.pt")  # 显示结果
        """ surface_shape_evaluator.update_our_surface(heights)
        surface_shape_evaluator.print_evaluation_summary(save_to_file=f"{cfg.output_foldername}/surface_summary2.txt")  # 打印评估结果
        surface_shape_evaluator.visualize_surfaces_comprehensive(save_path=f'{cfg.output_foldername}/shape_evaluate2/surface_comparison.png') """
        #make_obj(cfg.sample_size_for_obj, path_points_yz, x_grid, device=device, output_file=f"{cfg.output_foldername}/output.obj")  # 保存 OBJ 文件

from new_spline_generator import BSplinePathOptimizer
def train_with_bsplines(pretrained_pt=None):

    # 保存配置文件
    utils.save_config_to_file(cfg.output_foldername, "config.txt")
    utils.save_config_to_file(cfg.output_foldername, f"{cfg.extra_config_filename}.txt", readfilename=f"{cfg.extra_config_filename}.py")
    
    device = utils.cuda_init(cfg.cuda_num)
    Ps, init_path_points = init_with_camlib(device, use_x=True)  # 初始化采样点和路径点
    
    if pretrained_pt is not None:
        init_path_points = use_pt(pretrained_pt, device=device)
    
    print(f"原始路径形状: {init_path_points.shape}")
    path_num, path_len, _ = init_path_points.shape
    
    # 创建B样条优化器
    n_control = max(4, path_len // cfg.control_points_ratio)  # 控制点数量，至少4个
    print(f"使用 {n_control} 个控制点 (原路径长度: {path_len}, 压缩比: {n_control/path_len:.3f})")
    
    bspline_optimizer = BSplinePathOptimizer(
        path_num=path_num,
        n_control=n_control, 
        path_len=path_len,
        device=device
    )
    
    # 将初始路径拟合到B样条控制点
    bspline_optimizer.fit_to_initial_paths(init_path_points)
    
    # 验证拟合质量
    with torch.no_grad():
        reconstructed_paths = bspline_optimizer.evaluate_paths()
        fitting_error = torch.mean((reconstructed_paths - init_path_points) ** 2)
        print(f"B样条拟合误差: {fitting_error:.6f}")
        
        if fitting_error > 0.01:
            print("警告：拟合误差较大，建议增加控制点数量")
    
    # 准备可视化和渲染（使用重构后的路径）
    with torch.no_grad():
        fixed_receiver_points, fixed_heights = get_target_pos.get_target_pos(device=device)
        
        # 可视化初始B样条路径
        surface_points, dirs = get_dirs(Ps, reconstructed_paths)
        utils_draw.visualize_paths_and_surface(
            surface_points, reconstructed_paths, 
            grid_size=cfg.sample_size+1, 
            filename=f"{cfg.output_foldername}/surface_vis/bspline_paths_init.png", 
            path_sample_ratio=0.05, surface_alpha=0.3, path_colors=['red']
        )

        info_list, real_picture, target_picture, real_sobel_result, target_sobel_result = prepare_for_rendering(device, fixed_receiver_points)
        
        utils_draw.drawHeatMap(real_picture, f"{cfg.output_foldername}/real_picture.png", title="Real Picture")
        utils_draw.drawHeatMap(target_picture, f"{cfg.output_foldername}/target_picture.png", title="Target Picture")

        # 初始渲染结果
        receiver_points = get_receiver_points(Ps, reconstructed_paths)
        render_result = prepare_for_render.get_render_result(receiver_points, info_list)
        utils_draw.drawTwoHeatMap(target_picture, render_result, 
                                path=f'{cfg.output_foldername}/render_result/bspline_compare_-1.png', 
                                title1='Target', title2=f'B-Spline Render Result at init.')

    # 设置优化器：优化控制点而不是路径点
    start_learning_rate = cfg.learning_rate    
        
    optimizer = torch.optim.Adam(bspline_optimizer.get_parameters(), 
                                start_learning_rate, betas=(0.8, 0.95), eps=1e-8, amsgrad=False)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=50, cooldown=0, verbose=True, eps=1e-9)
    
    losses = []
    change_at_epoch = -1  # 不切换

    
 
    for epoch in range(cfg.max_epochs):
        optimizer.zero_grad()
        
        if epoch == change_at_epoch:
            optimizer.param_groups[0]['lr'] = cfg.learning_rate
            print("Switching to render loss optimization...")
        
        # 从控制点生成当前路径
        current_path_points = bspline_optimizer.evaluate_paths()  # [path_num, path_len, 3]
        
        if epoch < change_at_epoch:
            pass
        else:
            # 计算渲染损失
            render_loss, grad_to_path_points, receiver_points = get_render_loss(
                Ps, current_path_points, target_sobel_result, info_list
            )
        
        """ # B样条天然光滑，可以不用或减少正则化
        if cfg.use_bspline_regularization:
            # 对控制点进行轻量级正则化
            control_reg = cfg.control_reg_weight * torch.sum(bspline_optimizer.control_points ** 2)
            control_reg.backward() """
        
        # 处理CUDA手动梯度（如果有）
        if epoch >= change_at_epoch and grad_to_path_points is not None:
            # 将路径点梯度传播到控制点
            # 方法：使用虚拟损失触发反向传播
            dummy_loss = torch.sum(current_path_points * grad_to_path_points.detach())
            dummy_loss.backward()
        
        if epoch < change_at_epoch:
            pass
        else:
            if 0:
                pass
            else:
                loss = render_loss
        
        optimizer.step()
        
        if epoch >= change_at_epoch:
            losses.append(loss.item())
            scheduler.step(loss)
        
        # 打印和可视化
        if epoch % cfg.print_epochs == 0 or epoch == cfg.max_epochs - 1:
            if epoch < change_at_epoch:
                pass
            else:
                if 0:
                    print(f"Epoch {epoch + 1}/{cfg.max_epochs}, "
                          f"Render Loss: {render_loss.item():.6f}, "
                          #f"Control Reg: {control_reg.item():.6f}, "
                          f"Total Loss: {loss.item():.6f}")
                else:
                    print(f"Epoch {epoch + 1}/{cfg.max_epochs}, "
                          f"Render Loss: {render_loss.item():.6f}")
            
            current_lr = optimizer.param_groups[0]['lr']
            if current_lr < 1e-9:
                print(f"Learning rate too small: {current_lr}, stopping training.")
                with torch.no_grad():
                    final_paths = bspline_optimizer.evaluate_paths()
                    receiver_points = get_receiver_points(Ps, final_paths)
                    render_result = prepare_for_render.get_render_result(receiver_points, info_list)
                    utils_draw.drawTwoHeatMap(target_picture, render_result, 
                                            path=f'{cfg.output_foldername}/render_result/bspline_compare_{epoch}.png', 
                                            title1='Target', title2=f'B-Spline Result at epoch {epoch}')
                    break

        if (epoch >= 0 and epoch % cfg.draw_epochs == 0) or epoch == cfg.max_epochs - 1:
            with torch.no_grad():
                current_paths = bspline_optimizer.evaluate_paths()
                receiver_points = get_receiver_points(Ps, current_paths)
                render_result = prepare_for_render.get_render_result(receiver_points, info_list)
                surface_points, dirs = get_dirs(Ps, current_paths)
                
                utils_draw.visualize_surface_vectors_3d(
                    surface_points, dirs, grid_size=cfg.sample_size+1, 
                    filename=f"{cfg.output_foldername}/surface_vis/bspline_surface_vectors_{epoch}.png"
                )
                
                utils_draw.drawTwoHeatMap(target_picture, render_result, 
                                        path=f'{cfg.output_foldername}/render_result/bspline_compare_{epoch}.png', 
                                        title1='Target', title2=f'B-Spline Result at epoch {epoch}')

    # 保存结果
    with torch.no_grad():
        final_paths = bspline_optimizer.evaluate_paths()
        
        # 绘制loss曲线
        utils_draw.drawLossCurve(losses, f"{cfg.output_foldername}/bspline_losses.png")
        
        # 保存控制点和最终路径
        torch.save(bspline_optimizer.control_points, f"{cfg.output_foldername}/bspline_control_points.pt")
        torch.save(final_paths, f"{cfg.output_foldername}/bspline_path_points.pt")
        
        print(f"最终控制点形状: {bspline_optimizer.control_points.shape}")
        print(f"最终路径形状: {final_paths.shape}")
        
        heights, normals, _ = new_intersect.intersect(Ps, final_paths, cfg.R, cfg.zmax)
        print(f"heights: {heights.shape}, normals: {normals.shape}")
        
        surface_points, dirs = get_dirs(Ps, final_paths)
        utils_draw.visualize_paths_and_surface(
            surface_points, final_paths, 
            grid_size=cfg.sample_size+1, 
            filename=f"{cfg.output_foldername}/surface_vis/bspline_paths_final.png", 
            path_sample_ratio=0.05, surface_alpha=0.3, path_colors=['red']
        )
        
        # 显示最终结果
        just_show_result(pretrained_pt=f"{cfg.output_foldername}/bspline_path_points.pt")
    
    return bspline_optimizer

def get_obj():
    device = utils.cuda_init(cfg.cuda_num)
    path_points_yz = torch.load(f"{cfg.output_foldername}/path_points_yz.pt")
    x_grid = bvh.make_x_grid(cfg.path_size, cfg.sample_size, cfg.R, device=device)  # 预计算可能相交的路径点
    make_obj(cfg.sample_size_for_obj, path_points_yz, x_grid, device=device, output_file=f"{cfg.output_foldername}/output2.obj")  # 
    
if __name__ == "__main__":
    just_show_result(pretrained_pt="./output/render_600_2000_0.02_27/"+"path_points.pt")  # 显示结果large_600_2000_0.01_large_24
