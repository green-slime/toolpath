import torch
import torchvision.transforms as T
from utils import load_dinov2_model
from PIL import Image
from pathlib import Path

"""
先设置各参数
"""
# 项目绝对地址：
project_dir = '/data/wzr/2025/'
# 快速ot build 绝对地址：
where_is_otmap = '/data/wzr/otmap/build'
# 设置dinov2所用特征patch的高度和宽度：
patch_h = 50
patch_w = patch_h
batch_size = 1
# 所计算图像的路径：
#img_path = f'/data/wzr/2025/img/gray_circle.png'
output_folder_index = 6
img_path = f'/data/wzr/2025/img/einstein_200.png'
img_path = f'/data/wzr/2025/img/6.jpg'
img_path = f'/data/wzr/2025/img/ustc_400.png'
#img_path = f'/data/wzr/2025/img/lena_200.png'
#img_path = './img/white.png'
image_name = Path(img_path).stem  # 'zju', 不含扩展名
# 图像大小
img_size = 200
# 样条采样密度：
nu=nv=401
# 接收平面的高度：
z_of_receiver = 3.0
# 样条控制点密度
control_points_num = 100

"""
各初始化函数，尽量不动
"""
# 定义图像转换操作
transform = T.Compose([
    T.GaussianBlur(9, sigma=(0.1, 2.0)),  # 高斯模糊
    T.Resize((patch_h * 14, patch_w * 14)),  # 调整图像大小
    T.CenterCrop((patch_h * 14, patch_w * 14)),  # 中心裁剪
    T.ToTensor(),  # 转换为张量
    # 这里标准化的参数取决于数据的均值和标准差，下面的参数是 ImageNet-1k 的统计结果
    # 若换为其它的数据，则重新计算为好
    #T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # 标准化
])

transform2 = T.Compose([
    T.Resize((img_size,img_size)),
    T.CenterCrop((img_size, img_size)),
    T.ToTensor()
])

def cuda_init(gpu_index:int)->torch.device:
    """
    gpu_index: 用几号gpu
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = torch.device(f"cuda:{gpu_index}")
        torch.cuda.set_device(device)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device

def prepare(device):
    """
    准备patchtoken和real_picture
    """
    # 使用torch.hub加载dinov2_vits14模型并移至CUDA设备
    #dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
    dinov2_vits14 = load_dinov2_model('./models/dinov2', device)
    # 特征维度, vits:384, vitb:768, vitl:1024
    feat_dim = 384

    # 创建用于存储特征和图像张量的零张量
    imgs_tensor = torch.zeros(batch_size, 3, patch_h * 14, patch_w * 14).to(device)
    # 打开图像并转换为灰度模式
    img = Image.open(img_path).convert('L')
    #print(np.array(img).shape)
    # TODO: 先进行gamma矫正：
    #
    # 对图像进行转换操作，并将其存储在imgs_tensor的第一个位置
    pic = transform(img)
    #print(pic.shape)
    imgs_tensor[0][0] = pic
    #print(imgs_tensor[0])

    with torch.no_grad():
        # 将图像张量传递给dinov2_vits14模型获取特征
        features_dict = dinov2_vits14.forward_features(imgs_tensor)
        patchtokens = features_dict['x_norm_patchtokens']
        clstoken = features_dict['x_norm_clstoken']
        patchtoken = torch.mean(patchtokens, dim = 2).detach().to(device).requires_grad_(False)
        # patchtoken.shape = (batch_size, patch_h*patch_w)
    
        real_picture=transform2(img).squeeze().float().detach().to(device)
        # 这里预计算样条的 u,v,du,dv
         
     
    return patchtoken, real_picture

def only_prepare_img(img_path, device):
    """
    只准备图像
    """

    # 打开图像并转换为灰度模式
    img = Image.open(img_path).convert('L')
    #print(np.array(img).shape)
    real_picture=transform2(img).squeeze().float().detach().to(device)
    return real_picture