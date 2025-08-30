import torch
import config as cfg
from old_code.NURBS import NURBS
import utils
from render import trace_rays_through_surface

def get_target_pos(device='cpu'):
    """
    输出：
        receiver_points: [N², 2] 的张量，表示 fixed 目标落点。
    """
    print("Establishing target surface...")
    with torch.no_grad():
        old_project_path = '/data/wzr/2025'
        control_points1 = torch.load(f'{old_project_path}/{cfg.control_points_name}').to(device)
        wij = torch.load(f'{old_project_path}/{cfg.wij_name}').to(device)
        
        sample_points = utils.make_2d_sample_points(grid_size=cfg.sample_size, device=device)
        #print(sample_points)
        bsurface = NURBS(control_points1, degree_u=3, degree_v=3, sample_points=sample_points, flag_large_sample_size=True)
        heights, normals = bsurface.evaluate_batch(sample_points, control_points1, wij, batch_size=100000)
        #bsurface.save_to_obj(control_points1, wij, nu=cfg.sample_size+1, nv=cfg.sample_size+1, filename=f"{cfg.output_foldername}/surface.obj")

        surface_points = torch.cat((sample_points, heights.unsqueeze(-1)), dim=-1)  # [N, 3]
        
        receiver_points = trace_rays_through_surface(surface_points, normals, z_receiver=cfg.z_of_receiver, n1=cfg.n1, n2=cfg.n2) 
        
        return receiver_points.clone().detach().requires_grad_(False), heights.clone().detach().requires_grad_(False)

def get_target_pos2(device='cpu'):
    """
    输出：
        receiver_points: [N², 2] 的张量，表示 fixed 目标落点。
    """
    print("Establishing target surface...")
    with torch.no_grad():
        old_project_path = '/data/wzr/2025'
        control_points1 = torch.load(f'{old_project_path}/{cfg.control_points_name}').to(device)
        wij = torch.load(f'{old_project_path}/{cfg.wij_name}').to(device)
        
        sample_points = utils.make_2d_sample_points(grid_size=cfg.sample_size, device=device)
        #print(sample_points)
        bsurface = NURBS(control_points1, degree_u=3, degree_v=3, sample_points=sample_points, flag_large_sample_size=True)
        heights, normals = bsurface.evaluate_batch(sample_points, control_points1, wij, batch_size=500000)
        #bsurface.save_to_obj(control_points1, wij, nu=cfg.sample_size+1, nv=cfg.sample_size+1, filename=f"{cfg.output_foldername}/surface.obj")

        surface_points = torch.cat((sample_points, heights.unsqueeze(-1)), dim=-1)  # [N, 3]
        
        receiver_points = trace_rays_through_surface(surface_points, normals, z_receiver=cfg.z_of_receiver, n1=cfg.n1, n2=cfg.n2) 
        
        return receiver_points.clone().detach().requires_grad_(False), heights.clone().detach().requires_grad_(False), normals.clone().detach().requires_grad_(False)
    
if __name__ == "__main__":
    device = utils.cuda_init(0)
    old_project_path = '/data/wzr/2025'
    control_points1 = torch.load(f'{old_project_path}/{cfg.control_points_name}').to(device)
    wij = torch.load(f'{old_project_path}/{cfg.wij_name}').to(device)
    
    sample_points = utils.make_2d_sample_points(grid_size=200, device=device)
    #print(sample_points)
    bsurface = NURBS(control_points1, degree_u=3, degree_v=3, sample_points=sample_points)
    heights, normals = bsurface.evaluate_batch(sample_points, control_points1, wij, batch_size=500000)
    utils.save_to_obj(heights, normals, nu=201, nv=201, filename=f"{cfg.output_foldername}/surface.obj")