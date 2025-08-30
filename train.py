from batch_intersect_min import batch_intersect, batch_intersect_with_cuda, batch_intersect_for_render, batch_intersect_for_height
import bvh
import render
import torch
import utils
import config as cfg
from init import init
from init_with_camlib import init_with_camlib
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

def get_receiver_points(Ps, path_points_yzs, x_grid, device='cpu'):
    heights, normals, _, _ = batch_intersect_with_cuda(Ps, cfg.R, cfg.sample_size, x_grid, path_points_yzs, batch_size = cfg.batch_size, device=device, alpha=cfg.alpha, MAX_HEIGHT=cfg.zmax)
    Ps_xy = Ps[:, :, :2].reshape(-1, 2)  # 取出采样点的 x 和 y 分量
    surface_points = torch.cat([Ps_xy, heights.unsqueeze(-1)], dim=-1)  # 形状 (N, 3)，N 是采样点的总数
    receiver_points = render.trace_rays_through_surface(surface_points, normals, z_receiver=cfg.z_of_receiver, n1=cfg.n1, n2=cfg.n2)
    return receiver_points

def get_dirs(Ps, path_points_yzs, x_grid, device='cpu'):
    heights, normals, dirs, _, _ = batch_intersect_with_cuda(Ps, cfg.R, cfg.sample_size, x_grid, path_points_yzs, batch_size = cfg.batch_size, device=device, alpha=cfg.alpha, MAX_HEIGHT=cfg.zmax, need_dirs=True)
    Ps_xy = Ps[:, :, :2].reshape(-1, 2)  # 取出采样点的 x 和 y 分量
    surface_points = torch.cat([Ps_xy, heights.unsqueeze(-1)], dim=-1)  # 形状 (N, 3)，N 是采样点的总数
    
    return surface_points, dirs


def get_ot_loss(Ps, x_grid, path_points_yzs, fixed_receiver_points, device='cpu'):
    receiver_points = get_receiver_points(Ps, path_points_yzs, x_grid, device=device)  # 获取当前接收器点
    ot_loss = torch.norm(receiver_points - fixed_receiver_points, p=2) ** 2
    return ot_loss

def ot_loss_with_cuda(Ps, x_grid, path_points_yzs, fixed_receiver_points, device='cpu'):
    _, _, loss, grad = batch_intersect_with_cuda(Ps, cfg.R, cfg.sample_size, x_grid, path_points_yzs, batch_size=cfg.batch_size, device=device, alpha=cfg.alpha, MAX_HEIGHT=cfg.zmax, target_pos=fixed_receiver_points)
    return loss, grad

def height_loss_with_cuda(Ps, x_grid, path_points_yzs, target_height, device='cpu'):
    """
    计算高度损失，使用 CUDA 扩展。
    """
    _, _, loss, grad = batch_intersect_for_height(Ps, cfg.R, cfg.sample_size, x_grid, path_points_yzs, batch_size=cfg.batch_size, device=device, alpha=cfg.alpha, MAX_HEIGHT=cfg.zmax, target_height=target_height)
    return loss, grad

def get_render_loss(Ps, x_grid, path_points_yzs, sobel_result, infolist, device='cpu'):
    loss, grad, receiver_points = batch_intersect_for_render(Ps, cfg.R, cfg.sample_size, x_grid, path_points_yzs, sobel_result, infolist, batch_size=cfg.batch_size, device=device, MAX_HEIGHT=cfg.zmax)
    return loss, grad, receiver_points

def overlay_loss(path_points_yzs):
    """
    可以证明，当相邻路径点的 y 分量不超过 R 时，可以全覆盖。
    """
    ys = path_points_yzs[:, :, 0]  # 取出路径点的 y 分量，形状 (path_num, path_size + 1)
    loss1 = torch.sum(torch.relu(torch.abs(ys[1:, :] - ys[:-1, :]) - cfg.R))  # 相邻路径点的 y 分量差
    loss2 = torch.sum(torch.relu(torch.abs(ys[0,:]-0.0) - cfg.R))  # 第一个路径点的 y 分量
    loss3 = torch.sum(torch.relu(torch.abs(ys[-1,:]-1.0) - cfg.R))  # 最后一个路径点的 y 分量
    return loss1 + loss2 + loss3

def use_yz_pt(yz_pt_name, device='cpu'):
    """
    使用预训练的路径点，如果不存在则返回 None。
    """
    try:
        path_points_yz = torch.load(yz_pt_name, map_location=device)
        print(f"使用预训练的路径点: {yz_pt_name}")
        return path_points_yz.clone().detach().requires_grad_(True)  # 返回一个叶子节点
    except FileNotFoundError:
        print(f"未找到预训练的路径点: {yz_pt_name}")
        exit()
        
def make_obj(sample_size, path_points_yz, x_grid, device='cpu', output_file=cfg.obj_filename):
    from init import init_sample_points
    Ps = init_sample_points(sample_size, device=device)  # 初始化采样点
    x_grid = bvh.make_x_grid(cfg.path_size, sample_size, cfg.R, device=device)  # 预计算可能相交的路径点
    heights, normals, _, _ = batch_intersect_with_cuda(Ps, cfg.R, sample_size, x_grid, path_points_yz, batch_size = cfg.batch_size, device=device, alpha=cfg.alpha, MAX_HEIGHT=cfg.zmax)
    #heights, normals = batch_intersect(Ps, cfg.R, sample_size, x_grid, path_points_yz, batch_size = cfg.batch_size, device=device, alpha=cfg.alpha, MAX_HEIGHT=cfg.zmax)
    #print(normals.reshape(-1, sample_size +1, 3)[:,8:12,:])
    #exit()
    #utils.create_closed_grid(sample_size, heights, normals, min_height=None, output_file=output_file)
    utils.save_to_obj(heights, normals, nu=sample_size + 1, nv=sample_size + 1, filename=output_file)  # 保存 OBJ 文件
    
def mocked_fixed_target(device):
    Ps, _, path_points_yz = init(device)  # 初始化采样点和路径点
    path_points_yz = path_points_yz.clone().detach().requires_grad_(False)
    path_points_yz[:,:, 0] += 1e-3  # 模拟的路径点偏移
    x_grid = bvh.make_x_grid(cfg.path_size, cfg.sample_size, cfg.R, device=device)  # 预计算可能相交的路径点
    heights, normals = batch_intersect(Ps, cfg.R, cfg.sample_size, x_grid, path_points_yz, batch_size = cfg.batch_size, device=device, alpha=cfg.alpha, MAX_HEIGHT=cfg.zmax)
    Ps_xy = Ps[:, :, :2].reshape(-1, 2)  # 取出采样点的 x 和 y 分量
    surface_points = torch.cat([Ps_xy, heights.unsqueeze(-1)], dim=-1)  # 形状 (N, 3)，N 是采样点的总数
    receiver_points = render.trace_rays_through_surface(surface_points, normals, z_receiver=cfg.z_of_receiver, n1=cfg.n1, n2=cfg.n2)
    return receiver_points.clone().detach().requires_grad_(False)  

def train(pretrained_pt=None):
    device = utils.cuda_init(cfg.cuda_num)
    Ps, path_points_yz = init_with_camlib(device)  # 初始化采样点和路径点
    if pretrained_pt is not None:
        path_points_yz = use_yz_pt(pretrained_pt, device=device)

    
    #fixed_receiver_points = get_fixed_receiver_points(device=device)  # 获取固定的接收器点
    with torch.no_grad():
        fixed_receiver_points, fixed_heights = get_target_pos.get_target_pos(device=device)  # 获取目标接收器点
        #fixed_receiver_points = mocked_fixed_target(device)  # 使用模拟的固定接收器点

        x_grid = bvh.make_x_grid(cfg.path_size, cfg.sample_size, cfg.R, device=device)  # 预计算可能相交的路径点
        surface_points, dirs = get_dirs(Ps, path_points_yz, x_grid, device=device)  # 获取当前方向向量
        
        #print(f"target height 第一行: {fixed_heights.reshape(cfg.sample_size+1, cfg.sample_size+1)[:,0]}")
        #print(f"path_points_yz 第一行：{path_points_yz[0,:,1]}")
        #print(f"surface points 第一行： {surface_points.reshape(cfg.sample_size+1, cfg.sample_size+1, 3)[:,0,2]}")
        """ surface_shape_evaluator = evaluate_surface_shape.surface_shape_evaluator(device, Ps, fixed_heights, surface_points[:, 2])  # 创建曲面形状评估器
        surface_shape_evaluator.print_evaluation_summary(save_to_file=f"{cfg.output_foldername}/surface_summary1.txt")  # 打印评估结果
        surface_shape_evaluator.visualize_surfaces_comprehensive(save_path=f'{cfg.output_foldername}/shape_evaluate/surface_comparison.png')
        utils_draw.visualize_surface_vectors_3d(surface_points, dirs, grid_size=cfg.sample_size+1, filename=f"{cfg.output_foldername}/surface_vis/surface_vectors_-1.png")  # 绘制表面向量 """

        print("准备渲染所需的信息...")
        real_picture, real_sobel_result = prepare_for_render.prepare(device=device)
        info_list = prepare_for_render.get_neeeded_info(device=device, real_picture=real_picture)
        target_picture, target_sobel_result = prepare_for_render.prepare_target_img(fixed_receiver_points, info_list, device=device)  # 准备目标图像和 Sobel 结果
        prepare_for_render.modify_infolist(info_list, target_picture)  # 替换 infolist 中的 real_picture 为 target_picture
        utils_draw.drawHeatMap(real_picture, f"{cfg.output_foldername}/real_picture.png", title="Real Picture")  # 绘制真实图像
        utils_draw.drawHeatMap(target_picture, f"{cfg.output_foldername}/target_picture.png", title="Target Picture")  # 绘制真实图像

        receiver_points = get_receiver_points(Ps, path_points_yz, x_grid, device=device)
        render_result = prepare_for_render.get_render_result(receiver_points, info_list)
        utils_draw.drawTwoHeatMap(target_picture, render_result, path=f'{cfg.output_foldername}/render_result/compare_-1.png', title1='Target', title2=f'Render Result at init.')

        
        
        #receiver_points = get_receiver_points(Ps, path_points_yz, x_grid, device=device)

        #utils_draw.drawPairPos(receiver_points, fixed_receiver_points, epoch='init', filename=f"{cfg.pairpos_foldername}/init.png")
        #utils_draw.drawScatterPos(receiver_points, filename=f"{cfg.output_foldername}/init.png")  # 绘制接收器点
    if cfg.change_at_epoch < 0: # 不进行 ot 的前提下
        start_learning_rate = cfg.learning_rate
    else:
        start_learning_rate = cfg.ot_learning_rate
    optimizer = torch.optim.Adam([path_points_yz], start_learning_rate, betas=(0.8, 0.95), eps=1e-8, amsgrad=False)
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
    #change_at_epoch = -1  # 不切换
    change_at_epoch = cfg.change_at_epoch

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
            ot_loss, grad = ot_loss_with_cuda(Ps, x_grid, path_points_yz, fixed_receiver_points, device=device)
            receiver_points = get_receiver_points(Ps, path_points_yz, x_grid, device=device)  # 获取当前接收器点
        else:
            render_loss, grad, receiver_points = get_render_loss(Ps, x_grid, path_points_yz, target_sobel_result, info_list, device=device)
        #ot_loss, grad = ot_loss_with_cuda(Ps, x_grid, path_points_yz, fixed_receiver_points, device=device)
        if path_points_yz.grad is None:
            # 第一次迭代需要初始化梯度张量
            path_points_yz.grad = torch.zeros_like(path_points_yz)
        path_points_yz.grad.copy_(grad)
        #loss = ot_loss
        overlayloss = overlay_loss(path_points_yzs=path_points_yz) * cfg.overlay_weights
        overlayloss.backward() # 梯度自动累加
        #loss =  ot_loss + overlayloss * cfg.overlay_weights
        if epoch < change_at_epoch:
            loss = ot_loss + overlayloss   # 使用 OT 损失和覆盖损失的加权和
        else:
            loss = render_loss + overlayloss  # 使用渲染损失和覆盖损失的加权和
        #loss.backward()
        #print(f"Gradient: {path_points_yz.grad}")  # 打印梯度
        optimizer.step()
        losses.append(loss.item())
        scheduler.step(loss)  # 更新学习率
        if epoch % cfg.print_epochs == 0 or epoch == cfg.max_epochs - 1:
            #print(f"Epoch {epoch + 1}/{cfg.max_epochs}, Loss: {loss.item():.6f}")
            if epoch < change_at_epoch:
                print(f"Epoch {epoch + 1}/{cfg.max_epochs}, Loss: {loss.item():.6f}, OT Loss: {ot_loss.item():.6f}, Overlay Loss: {overlayloss.item():.6f}")
            else:
                print(f"Epoch {epoch + 1}/{cfg.max_epochs}, Loss: {loss.item():.6f}, render Loss: {render_loss.item():.6f}, Overlay Loss: {overlayloss.item():.6f}")
            #print(f"Epoch {epoch + 1}/{cfg.max_epochs}, Loss: {loss.item():.6f}, OT Loss: {ot_loss.item():.6f}, Overlay Loss: {overlayloss.item():.6f}")
            #exit()
            current_lr = optimizer.param_groups[0]['lr']
            if(current_lr < 1e-9):
                print(f"Learning rate too small: {current_lr}, stopping training.")
                with torch.no_grad():
                    #receiver_points = get_receiver_points(Ps, path_points_yz, x_grid, device=device)  # 获取当前接收器点
                    render_result = prepare_for_render.get_render_result(receiver_points, info_list)
                    #utils_draw.drawHeatMap(render_result, f"{cfg.output_foldername}/render_result/render_result_{epoch}.png", title=f"Render Result at Epoch {epoch}")  
                    utils_draw.drawTwoHeatMap(target_picture, render_result, path=f'{cfg.output_foldername}/render_result/compare_{epoch}.png', title1='Target', title2=f'Render Result at epoch {epoch}')
                    break
        #if epoch % cfg.draw_epochs == 0 or epoch == cfg.max_epochs - 1:
            #print(f"Epoch {epoch + 1}/{cfg.max_epochs}, Path Points YZ: {path_points_yz}")
        if (epoch >= 0 and epoch % cfg.draw_epochs == 0) or epoch == cfg.max_epochs - 1:
            with torch.no_grad():
                #receiver_points = get_receiver_points(Ps, path_points_yz, x_grid, device=device)  # 获取当前接收器点
                render_result = prepare_for_render.get_render_result(receiver_points, info_list)
                #surface_points, dirs = get_dirs(Ps, path_points_yz, x_grid, device=device)  # 获取当前方向向量
                #utils_draw.visualize_surface_vectors_3d(surface_points, dirs, grid_size=cfg.sample_size+1, filename=f"{cfg.output_foldername}/surface_vis/surface_vectors_{epoch}.png")  # 绘制表面向量
                #utils_draw.drawHeatMap(render_result, f"{cfg.output_foldername}/render_result/render_result_{epoch}.png", title=f"Render Result at Epoch {epoch}")
                utils_draw.drawTwoHeatMap(target_picture, render_result, path=f'{cfg.output_foldername}/render_result/compare_{epoch}.png', title1='Target', title2=f'Render Result at epoch {epoch}')  # 绘制目标图像和渲染结果的对比图
    with torch.no_grad():
        # 绘制 loss 曲线
        utils_draw.drawLossCurve(losses, cfg.losses_filename)
        # 保存最终的路径点
        #print(path_points_yz)
        heights, normals, _, _ = batch_intersect_with_cuda(Ps, cfg.R, cfg.sample_size, x_grid, path_points_yz, batch_size = cfg.batch_size, device=device, alpha=cfg.alpha, MAX_HEIGHT=cfg.zmax)
        print(f"heights:{heights}, normals:{normals}")
        torch.save(path_points_yz, f"{cfg.output_foldername}/path_points_yz.pt")  # 保存路径点
        """ surface_shape_evaluator.update_our_surface(heights)
        surface_shape_evaluator.print_evaluation_summary(save_to_file=f"{cfg.output_foldername}/surface_summary2.txt")  # 打印评估结果
        surface_shape_evaluator.visualize_surfaces_comprehensive(save_path=f'{cfg.output_foldername}/shape_evaluate2/surface_comparison.png') """
        #make_obj(cfg.sample_size_for_obj, path_points_yz, x_grid, device=device, output_file=f"{cfg.output_foldername}/output.obj")  # 保存 OBJ 文件
        
def get_obj():
    device = utils.cuda_init(cfg.cuda_num)
    path_points_yz = torch.load(f"{cfg.output_foldername}/path_points_yz.pt")
    x_grid = bvh.make_x_grid(cfg.path_size, cfg.sample_size, cfg.R, device=device)  # 预计算可能相交的路径点
    make_obj(cfg.sample_size_for_obj, path_points_yz, x_grid, device=device, output_file=f"{cfg.output_foldername}/output2.obj")  # 
    
if __name__ == "__main__":
    #train()
    foldername = "output/large_300_1000_0.01_large_15"
    device = utils.cuda_init(cfg.cuda_num)
    #path_points_yz = use_yz_pt(f"{foldername}/path_points_yz.pt", device=device)
    Ps, path_points_yz = init_with_camlib(device)  # 初始化采样点和路径点
    
    x_grid = bvh.make_x_grid(cfg.path_size, cfg.sample_size, cfg.R, device=device)  # 预计算可能相交的路径点
    make_obj(cfg.sample_size, path_points_yz, x_grid, device=device, output_file=f'{foldername}/output_init.obj')  # 保存 OBJ 文件
