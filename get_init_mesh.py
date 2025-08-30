import get_init_pos
import config as cfg
import utils

def get_init_mesh(sample_num=200, device='cpu', filename='./init_mesh.obj'):
    x_num = sample_num
    y_num = sample_num
    sample_points = get_init_pos.generate_sample_points(x_num, y_num, device=device)
    #print(sample_points, sample_points.shape)
    surface_points, normals = get_init_pos.get_nurbs_data(sample_points, device)
    utils.save_to_obj(surface_points[:,2], normals, nu=x_num, nv=y_num, filename=filename)
    
if __name__ == "__main__":
    device = utils.cuda_init(0)
    get_init_mesh(device)
    print("Initial mesh saved as 'init_mesh.obj'.")