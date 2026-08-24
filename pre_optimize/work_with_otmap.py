import subprocess
import os
from pathlib import Path
import utils
import config as cfg

where_is_otmap = '../otmap/build'

process_img_size = cfg.nu - 1
process_img_name = Path(cfg.img_path).stem  # 'zju'
img_folder_path = os.path.dirname(cfg.img_path) # './img'

def run_otmap(img_path = cfg.img_path, out_prefix = f'{img_folder_path}/{process_img_name}'):
    # 完整的 out_path: {out_prefix}_vectors.bin
    original_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(where_is_otmap)
    command = f'./otmap -in {img_path} -out {out_prefix}'
    
    try:
        # 运行命令并等待完成
        subprocess.run(command, shell=True, check=True)
        print("ot命令执行成功")
    except subprocess.CalledProcessError as e:
        print(f"ot命令执行失败，错误信息: {e}")
    finally:
        # 无论成功与否，切换回代码文件所在目录
        os.chdir(original_dir)
        
def resize_img_and_ot(img_path = cfg.img_path, out_prefix = f'{process_img_name}_{process_img_size}'):
    output_path = utils.resize_and_save_image(img_path, size=process_img_size)
    run_otmap(output_path, f'{cfg.project_dir}/otmap/'+out_prefix)
    


if __name__ == "__main__":
    resize_img_and_ot(out_prefix=f'{process_img_name}_{process_img_size}')
    a = utils.read_vectors_from_binary(f'{cfg.project_dir}/otmap/{process_img_name}_{process_img_size}_vectors.bin')
    print(a.shape)