import train
import train_with_normal
import train_with_height
import new_height_train
import new_render_train
import config as cfg
import os
import math
import datetime

def update_output_config(sample_size, output_foldername):
    cfg.sample_size = sample_size
    cfg.output_foldername = output_foldername
    cfg.temp_triangles_filename = f"temp/triangles_{sample_size}.pt"  # 临时三角形数据文件
    cfg.losses_filename = f"{output_foldername}/losses.png"  # 损失记录文件
    cfg.pairpos_foldername = f"{output_foldername}/pair_pos/" # 可视化投影结果
    cfg.obj_filename = f"{output_foldername}/output.obj"  # 输出的 OBJ 文件
    
def write_experiment_note(note, filename="experiment_note.txt"):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cuda_name = f"CUDA_{cfg.cuda_num}" if hasattr(cfg, 'cuda_num') else "CUDA_Unknown"
    with open(filename, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] [cuda: {cuda_name}] {note}\n")

if __name__ == "__main__":
    cfg.extra_config_filename = os.path.basename(os.path.splitext(__file__)[0])  # 额外配置文件名

    note = f"{cfg.img_name}:\n"
    write_experiment_note(note)

    cfg.max_epochs = 1000
    cfg.cuda_num = 2
    version = 3
    train_type = "render"
    cfg.overlay_weights = 1000.0
    cfg.learning_rate = 2e-6
    cfg.sobel_scale = 0.1
    cfg.l1_scale = 0.2
    #cfg.path_num = 800
    #cfg.control_points_ratio = 2
    cfg.reg_weight = 0.0  # 正则约束损失权重
    large_sample_size = cfg.sample_size
    #cfg.sample_size = 1000
    cfg.output_foldername = f"new_output/{train_type}_{cfg.ocl_path_num}_{cfg.sample_size}_{cfg.R}_{version}/"
    update_output_config(cfg.sample_size, cfg.output_foldername)
    new_render_train.train(pretrained_pt=None)
    #new_render_train.train_with_bsplines(pretrained_pt=None)

    note2 = f"训练完成。相关输出文件夹为：\n" \
            f"{cfg.output_foldername}\n"
    write_experiment_note(note2)

       
    