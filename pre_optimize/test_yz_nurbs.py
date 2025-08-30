"""
此test专注于解决从落点优化控制点高度场的问题。
"""
"""
这里发现简单地用图像loss+提高采样点个数（相应地，需要提高控制点个数）就可以了，效果有很大改善。
"""

import torch
import render
import torch.optim as optim
import utils
import config as cfg
import cuda_extension
import torch.nn.functional as F
from PIL import Image
import work_with_otmap 

device = cfg.cuda_init(0)
#patchtoken, real_picture = cfg.prepare(device)
real_picture = cfg.only_prepare_img(cfg.img_path, device)
info_list = render.get_neeeded_info(device, cfg.nu, cfg.nv, image_size=640, real_picture=real_picture, control_points_num=cfg.control_points_num, using_nurbs=True)  
z_of_receiver = cfg.z_of_receiver
cp_num = cfg.control_points_num

control_points_standard = torch.ones((cp_num,cp_num), device=device, requires_grad=True)*0.01

real_sobel_result = cuda_extension.real_sobel(
    real_picture,
    cfg.img_size
)

run_otmap_first = True
if(run_otmap_first):
    work_with_otmap.resize_img_and_ot()

#bin_name = "/data/wzr/2025/otmap/einstein_200_vectors.bin"
#bin_name = "/data/wzr/2025/otmap/gray_circle_200_vectors.bin"
bin_name = "/data/wzr/2025/otmap/lena_200_vectors.bin"
bin_name = "/data/wzr/2025/otmap/einstein_200_400_vectors.bin" 
bin_name = "/data/wzr/2025/otmap/6_400_vectors.bin" 
bin_name = "/data/wzr/2025/otmap/ustc_400_vectors.bin"
bin_name = "/data/wzr/2025/otmap/lena_200_400_vectors.bin"
bin_name = "/data/wzr/2025/otmap/ustc_400_400_vectors.bin"
version = 9

tensor = utils.read_vectors_from_binary(bin_name).to(device).requires_grad_(False)
def visualize_tensor_result():
    render_result = cuda_extension.render(
                    info_list[1],
                    info_list[2],
                    tensor.clone().detach(),
                    info_list[3],
                    info_list[4],
                    cfg.img_size
                )[0]
    
    utils.drawHeatMap(render_result, path=f'./results_yz_nurbs/ot_sample.png', title=f'yz render result: ot_sample')
                
fixed_normals = render.compute_target_normal(tensor, info_list, z_of_receiver).requires_grad_(False)

DRAW_EPOCH = 2000
PRINT_EPOCH = 10
MAX_EPOCH = 1500
def y2z_test(proj_xy_fixed, control_points, info_list, z_of_receiver):
    control_points1 = control_points.clone().detach().requires_grad_(True) # 变量
    control_points1 = torch.load('control_points7.pth').to(device).requires_grad_(True) # 变量
    wij = torch.ones((cp_num,cp_num), device=device, requires_grad=True) # 变量2
    wij = torch.load('wij7.pth').to(device).requires_grad_(True) # 变量2
    # wij 将过一个 ReLU 来保持其非负性
    proj_xy_fixed1 = proj_xy_fixed.clone().detach().requires_grad_(False) # 常数
    optimizer_z = torch.optim.Adam([control_points1], lr=2e-5, weight_decay=1e-3, amsgrad=True)
    optimizer_wij = torch.optim.Adam([wij], lr=1e-3, weight_decay=1e-3, amsgrad=True)
    #scheduler = optim.lr_scheduler.StepLR(optimizer_z, step_size=200, gamma=0.5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_z, mode='min', factor=0.6, patience=10, cooldown=10, verbose=True)
    losses = []
    max_epoch = MAX_EPOCH
    
    for epoch in range(max_epoch):
        wij1 = torch.nn.functional.relu(wij)
        proj_xy = render.get_weighted_proj_xy(info_list, info_list[6], control_points1, wij1, z_of_receiver)
        #lossf = useloss1(proj_xy, proj_xy_fixed1)
        #lossf = useloss_ot(control_points1, wij1, info_list).sum()
        lossf, render_diff_tensor = useloss2(proj_xy, info_list)

        optimizer_z.zero_grad()
        optimizer_wij.zero_grad()
        lossf.backward()
        optimizer_z.step()
        optimizer_wij.step()
        #scheduler.step() 
        #loss = lossf.item()
        loss = torch.norm(render_diff_tensor, p=2).item()
        if epoch > 5:
            scheduler.step(loss)  # warm up
        with torch.no_grad():
            if (epoch+1) % DRAW_EPOCH == 0:
                #utils.drawDelaunyTriangulation(proj_xy, info_list[2], epoch, folder='triangulation_yz_nurbs')
                #utils.drawPairPos(proj_xy, proj_xy_fixed1, epoch, folder='yz')
                render_result = cuda_extension.render(
                    info_list[1],
                    info_list[2],
                    proj_xy.clone().detach(),
                    info_list[3],
                    info_list[4],
                    cfg.img_size
                )[0]
    
                utils.drawHeatMap(render_result, path=f'./results_yz_nurbs/{epoch+1}.png', title=f'yz render result: epoch={epoch+1}')
                
            if epoch % PRINT_EPOCH == 0:
                print(f'Epoch [{epoch+1}/{max_epoch}], Loss: {loss:.4f}')
            losses.append(loss)
    utils.drawLossCurve(losses, path='y2z_nurbs_loss.png')
    torch.save(control_points1, f'control_points{version}.pth')
    torch.save(wij, f'wij{version}.pth')
    print(f'control_points:{control_points1}')
    print(f'wij:{wij}')
    #utils.drawPairPos(proj_xy, proj_xy_fixed1, -1, folder='yz')
    return control_points1

def useloss1(proj_xy, proj_xy_fixed):
    """
    用于计算 proj_xy 和 proj_xy_fixed 之间的 loss
    """
    proj_diff = proj_xy-proj_xy_fixed
    #lossf = torch.abs(proj_diff).sum()
    lossf = torch.norm(proj_diff, p=2)
    return lossf

def useloss2(proj_xy, info_list):
    """
    用于计算 proj_xy 渲染的图像与真实图像之间的 loss
    """
    diff_result = cuda_extension.diff(
                    info_list[1],
                    info_list[2],
                    proj_xy.clone().detach(),
                    info_list[3],
                    info_list[5],
                    real_sobel_result[0],
                    real_sobel_result[1],
                    info_list[4],
                    cfg.img_size
                )
    scale = torch.norm(diff_result[1], p=2).sum()/torch.norm(diff_result[4], p=2).sum()
    #lossf1 = torch.abs(proj_xy-proj_xy_fixed).sum()
    points_diff = (diff_result[1]+0.0*diff_result[4]).detach().requires_grad_(False) # this is \par{loss}/\par{yi}, which should be seen as a constant
    #print(points_diff)
    #print(proj_xy)
    f = (points_diff * proj_xy).sum()
    
    render_diff_tensor = (diff_result[3]-info_list[5]).clone().detach() 

    return f, render_diff_tensor # \par{loss}/\par{z_i}

    
    #if epoch % PRINT_EPOCH == 0:
    #    print(f'lossf1={lossf1}, lossf2={lossf2}')
    #lossf =  lossf1 + lossf2
    return lossf2

def useloss_ot(control_points, wij, info_list):
    surface_heights, surface_normals = info_list[6].evaluate_vector2(info_list[0], control_points, wij, using_cache=True)
    normals = surface_normals[:,:2] / surface_normals[:,2:3]
    
    return torch.norm(normals - fixed_normals, p=2)

def additional_loss(proj_xy):
    xmin = 0; xmax = 1; ymin = 0; ymax = 1
    return F.relu(proj_xy[:,0]-xmax)+F.relu(proj_xy[:,1]-ymax)+F.relu(xmin-proj_xy[:,0])+F.relu(ymin-proj_xy[:,1])

def test_mocked_proj(max_height=0.005):
    """
    从随机控制点生成 proj_xy，然后反求这些控制点
    """
    control_points = torch.rand((cp_num, cp_num), device=device, requires_grad=False) * max_height + 0.01
    wij = torch.rand((cp_num, cp_num), device=device, requires_grad=False) * max_height + 1
    proj_xy = render.get_weighted_proj_xy(info_list, info_list[6], control_points, wij, z_of_receiver)
    utils.drawDelaunyTriangulation(proj_xy,info_list[2],-1,'mocked_triangulation')
    control_points1 = y2z_test(proj_xy, control_points_standard, info_list, z_of_receiver)
    
import numpy as np
def test_designed_img(max_height = 0.005):
    """
    todo: 如何解决生成的光强分布数组超过1的部分被截断的问题
    直接归一化试试
    """
    with torch.no_grad():
        xs = ys = torch.linspace(0, 1, cp_num)
        x_grid, y_grid = torch.meshgrid(xs, ys)
        xy_data = torch.tensor(
            np.stack([x_grid.flatten(), y_grid.flatten()], axis=1),
            dtype=torch.float32,
            device=device
        )
        control_points = 0.01 + max_height * (1-2*((xy_data[:,0]-0.5)**2+(xy_data[:,1]-0.5)**2)).view((cp_num, cp_num)) # 中间凸, 0-max_height
        wij = torch.ones((cp_num,cp_num), device=device)
        proj_xy = render.get_weighted_proj_xy(info_list, info_list[6], control_points, wij, z_of_receiver)
        render_result = cuda_extension.render(
                    info_list[1],
                    info_list[2],
                    proj_xy.clone().detach(),
                    info_list[3],
                    info_list[4],
                    cfg.img_size
                )[0]
        print(render_result.sum(), real_picture.sum())
        utils.drawHeatMap(render_result, path=f'./rendered_designed_img.png', title=f'designed_img render result')
        render_result = render_result / torch.max(render_result)
        # 转成图像保存
        tensor_img = render_result * 255
        # 转为 uint8 类型
        tensor_img = tensor_img.byte()  # 或 .to(torch.uint8)
        # 步骤 3：转换为 NumPy 数组
        img_array = tensor_img.cpu().numpy()
        # 步骤 4：保存为灰度图
        img = Image.fromarray(img_array, mode='L')  # 'L' 表示灰度图
        img.save('designed_img.png')  # 保存为 PNG 文件
        info_list[6].save_to_obj(control_points, wij, cfg.nu, cfg.nv, filename = 'designed_surface.obj')
        
    
#test_mocked_proj(0.01) # test_mocked_proj 总是能够通过的。

# 更进一步地，让mocked_proj_xy渲染图像，然后由图像反求控制点。
# 这里需要更改一下info_list.

def test_mocked_img(max_height=0.005):
    """
    从随机控制点生成 proj_xy，然后反求这些控制点
    """
    control_points = torch.rand((cp_num, cp_num), device=device, requires_grad=False) * max_height + 0.01
    wij = torch.rand((cp_num, cp_num), device=device, requires_grad=False) * max_height + 1
    proj_xy = render.get_weighted_proj_xy(info_list, info_list[6], control_points, wij, z_of_receiver)
    render_result = cuda_extension.render(
                    info_list[1],
                    info_list[2],
                    proj_xy.clone().detach(),
                    info_list[3],
                    info_list[4],
                    cfg.img_size
                )[0]
    utils.drawHeatMap(render_result, path=f'./mock_img.png', title=f'mock_img render result')
    info_list1=info_list.copy()
    info_list1[5] = render_result
    info_list1[3] = render_result.sum().to(device)
    control_points1 = y2z_test(proj_xy, control_points_standard, info_list1, z_of_receiver)
    torch.save(control_points1, 'mock_control_points1.pth')
    utils.draw_2heights(control_points, control_points1, -1)
    visualize_diff_from_control_points(info_list1, z_of_receiver, control_points1)


def visualize_diff_from_control_points(info_list, z_of_receiver, control_points=None, wij=None):
    if control_points is None:
        control_points = torch.load('control_points1.pth')
    if wij is None:
        wij = torch.load('wij.pth')
    #print(control_points, wij)
    proj_xy = render.get_weighted_proj_xy(info_list, info_list[6], control_points, wij, z_of_receiver)
    render_tensor_result = cuda_extension.render(
                    info_list[1],
                    info_list[2],
                    tensor.clone().detach(),
                    info_list[3],
                    info_list[4],
                    cfg.img_size
                )[0]
    render_result = cuda_extension.render(
                    info_list[1],
                    info_list[2],
                    proj_xy.clone().detach(),
                    info_list[3],
                    info_list[4],
                    cfg.img_size
                )[0]
    utils.drawHeatMap(render_result - render_tensor_result, path=f'./diff_img.png', title=f'diff_img render result')
    print(f"diff loss:{torch.norm(render_result - render_tensor_result, p=1).sum()}")
    info_list[6].save_to_obj(control_points, wij, cfg.nu, cfg.nv, f'surface.obj')

def print_cp_and_wij():
    control_points1 = torch.load('control_points1.pth')
    wij = torch.load('wij.pth')
    print(control_points1)
    print(wij)

def calculate_maxmin_height():
    """
    计算控制点的最大最小高度
    """
    control_points = torch.load('control_points2.pth')
    wij = torch.load('wij2.pth')
    surface_heights, _ = info_list[6].evaluate_vector2(info_list[0], control_points, wij, using_cache=True)
    print(f"max:{surface_heights.max()}, min:{surface_heights.min()}")
    print(f"surface max-min:{surface_heights.max()- surface_heights.min()}, z0={cfg.z_of_receiver}")


#visualize_diff_from_control_points(info_list, z_of_receiver);exit() 
    
#test_mocked_img();exit()
"""
现在要考虑，如果真实解不存在，怎样的control_points是最好的？
"""
# 我们先再尝试一下直接用一范数当loss。
#proj_xy_fixed = torch.load('proj_xy1.pt')
# 可视化之
#utils.drawDelaunyTriangulation(proj_xy_fixed,info_list[2],-1,'triangulation_fixed')
# 观察直接优化的结果
#proj_xy_fixed = render.get_proj_xy(info_list, info_list[6], control_points_standard, z_of_receiver).detach()
""" control_points1 = torch.load('control_points1.pth')
proj_xy = render.get_proj_xy(info_list, info_list[6], control_points1, z_of_receiver)
utils.visualize_points(proj_xy, cfg.nu);exit(); """

def export_for_toolpath():
    """
    导出控制点和wij以供后续工具路径生成
    """
    info_list = render.get_neeeded_info(device, 51, 51, image_size=640, real_picture=real_picture, control_points_num=cfg.control_points_num, using_nurbs=True)  
    control_points1 = torch.load('control_points1.pth')
    wij = torch.load('wij1.pth')
    heights, normals = info_list[6].evaluate_vector2(info_list[0], control_points1, wij, using_cache=True)
    surface_points = torch.cat([info_list[0], heights.unsqueeze(1)], dim=1)
    proj_xy = render.get_weighted_proj_xy(info_list, info_list[6], control_points1, wij, z_of_receiver)
    torch.save(proj_xy, 'proj_xy1.pth')
    torch.save(surface_points, 'surface_points1.pth')  
    
proj_xy_fixed = utils.read_vectors_from_binary(bin_name).to(device).requires_grad_(False)
control_points1 = y2z_test(proj_xy_fixed, control_points_standard, info_list, z_of_receiver)
#export_for_toolpath()
exit()
#test_designed_img()
# 观察加入图像 loss 的结果: 与直接优化类似
# 观察直接以图像 loss 优化的结果
visualize_diff_from_control_points(info_list, z_of_receiver, torch.load('control_points1.pth'), torch.load('wij1.pth'))
#visualize_diff_from_control_points(info_list, z_of_receiver)
#visualize_tensor_result()


