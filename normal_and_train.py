import train
import train_with_normal
import config as cfg
import os
import math

def update_output_config(sample_size, output_foldername):
    cfg.sample_size = sample_size
    cfg.output_foldername = output_foldername
    cfg.temp_triangles_filename = f"temp/triangles_{sample_size}.pt"  # 临时三角形数据文件
    cfg.losses_filename = f"{output_foldername}/losses.png"  # 损失记录文件
    cfg.pairpos_foldername = f"{output_foldername}/pair_pos/" # 可视化投影结果
    cfg.obj_filename = f"{output_foldername}/output.obj"  # 输出的 OBJ 文件
    
if __name__ == "__main__":
    cfg.extra_config_filename = os.path.basename(os.path.splitext(__file__)[0])  # 额外配置文件名

    # normal train
    cfg.cuda_num = 2
    cfg.max_epochs = 500
    cfg.sample_size = 3000
    cfg.overlay_weights = 10000.0
    version_normal =  8
    ot_foldername = cfg.output_foldername = f"output/large_{cfg.ocl_path_num}_{cfg.sample_size}_0.01_normal_{version_normal}/"
    update_output_config(cfg.sample_size, cfg.output_foldername)
    train_with_normal.train()

    cfg.max_epochs = 1000
    version = 10
    cfg.overlay_weights = 1000.0
    large_sample_size = cfg.sample_size
    cfg.sample_size = 3000
    cfg.output_foldername = f"output/large_{cfg.ocl_path_num}_{cfg.sample_size}_0.01_large_{version}/"
    update_output_config(cfg.sample_size, cfg.output_foldername)
    train.train(pretrained_pt=ot_foldername+"/path_points_yz_normal_499.pt")
    exit()
       
    # small size render train   
    version_small = 6
    cfg.need_toolpath_preprocess = False
    cfg.sample_size = 500
    cfg.max_epochs = 300
    small_foldername = cfg.output_foldername = f"output/large_{cfg.ocl_path_num}_{cfg.sample_size}_0.01_small_{version_small}/"
    update_output_config(cfg.sample_size, cfg.output_foldername)
    train.train(pretrained_pt=ot_foldername+"/path_points_yz_ot.pt")
    
    # large size render train
    version = 9
    small_sample_size = cfg.sample_size
    cfg.sample_size = math.ceil(cfg.sample_size * math.sqrt(6))
    cfg.max_epochs = 400
    large_foldername = cfg.output_foldername = f"output/large_{cfg.ocl_path_num}_{cfg.sample_size}_0.01_large_{version}/"
    update_output_config(cfg.sample_size, cfg.output_foldername)
    train.train(pretrained_pt=small_foldername+"/path_points_yz.pt")
    #train.train(pretrained_pt=f"output/large_{cfg.ocl_path_num}_{small_sample_size}_0.01_ot_{version_ot}/path_points_yz_ot.pt")

    """ # large size render train again
    cfg.max_epochs = 600
    cfg.learning_rate = cfg.learning_rate * 0.5
    large_again_foldername = cfg.output_foldername = f"output/large_{cfg.ocl_path_num}_{cfg.sample_size}_0.01_large_{version}_again/"
    update_output_config(cfg.sample_size, cfg.output_foldername)
    train.train(pretrained_pt=large_foldername+"/path_points_yz.pt") """
    
    """ cfg.max_epochs = 1000
    cfg.output_foldername = f"output/large_{cfg.ocl_path_num}_{cfg.sample_size}_0.01_large_{version}_again_and_again/"
    update_output_config(cfg.sample_size, cfg.output_foldername)
    train.train(pretrained_pt=f"output/large_{cfg.ocl_path_num}_{cfg.sample_size}_0.01_large_{version}_again/path_points_yz.pt") """
    
    cfg.max_epochs = 500
    large_sample_size = cfg.sample_size
    cfg.sample_size = 3000
    cfg.output_foldername = f"output/large_{cfg.ocl_path_num}_{cfg.sample_size}_0.01_large_{version}_larger/"
    update_output_config(cfg.sample_size, cfg.output_foldername)
    train.train(pretrained_pt=large_foldername+"/path_points_yz.pt")
    
    
    
    
    
    # 生成输出文件夹信息文本
    output_folders_info = (
        "本次输出文件夹共有：\n"
        f"ot train: {ot_foldername}\n"
        f"small size render train: {small_foldername}\n"
        f"large size render train: {large_foldername}\n"
        #f"large size render train again: {large_again_foldername}\n"
        f"larger size render train: {cfg.output_foldername}\n"
    )

    print(output_folders_info)

    # 写入txt文件
    os.makedirs(cfg.output_foldername, exist_ok=True)
    summary_txt = os.path.join(cfg.output_foldername, "output_folders.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(output_folders_info)