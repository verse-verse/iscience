import os
import torch
import torch_directml  # 添加DirectML支持
import numpy as np
from osgeo import gdal
from models import UNet
import torch.nn.functional as F
from tqdm import tqdm

def load_model(model_path, device):
    """加载训练好的模型"""
    # 先将模型加载到CPU
    state_dict = torch.load(model_path, map_location='cpu')
    
    # 创建模型实例
    model = UNet(in_channels=4, out_channels=2)
    # 加载权重
    model.load_state_dict(state_dict)
    # 将模型移到目标设备
    model = model.to(device)
    model.eval()
    return model

def create_patches(image, patch_size=256, stride=256):
    """生成图像块的坐标"""
    # 确保输入图像是numpy数组
    if torch.is_tensor(image):
        image = image.cpu().numpy()
    
    h, w = image.shape[1:]  # image shape: (C, H, W)
    patches = []
    positions = []
    
    for y in range(0, h-patch_size+1, stride):
        for x in range(0, w-patch_size+1, stride):
            patch = image[:, y:y+patch_size, x:x+patch_size]
            patches.append(patch)
            positions.append((y, x))
    
    # 处理边缘情况
    # 右边缘
    if w % patch_size != 0:
        for y in range(0, h-patch_size+1, stride):
            x = w - patch_size
            patch = image[:, y:y+patch_size, x:x+patch_size]
            patches.append(patch)
            positions.append((y, x))
    
    # 下边缘
    if h % patch_size != 0:
        for x in range(0, w-patch_size+1, stride):
            y = h - patch_size
            patch = image[:, y:y+patch_size, x:x+patch_size]
            patches.append(patch)
            positions.append((y, x))
    
    # 右下角
    if w % patch_size != 0 and h % patch_size != 0:
        patch = image[:, h-patch_size:h, w-patch_size:w]
        patches.append(patch)
        positions.append((h-patch_size, w-patch_size))
    
    return patches, positions

def predict_large_image(input_path, output_path, model_path, device='cuda', confidence_threshold=0.5):
    """对大图进行预测"""
    # 打开输入图像
    ds = gdal.Open(input_path)
    if ds is None:
        raise RuntimeError(f"无法打开图像: {input_path}")
    
    # 获取图像信息
    width = ds.RasterXSize
    height = ds.RasterYSize
    proj = ds.GetProjection()
    geotrans = ds.GetGeoTransform()
    
    # 读取所有波段
    image = []
    for band in range(1, 5):
        band_data = ds.GetRasterBand(band).ReadAsArray()
        image.append(band_data)
    image = np.array(image)
    
    # 加载模型
    model = load_model(model_path, device)
    
    # 生成图像块
    patches, positions = create_patches(image)
    
    # 创建两个输出数组：一个存储概率值，一个存储最终结果
    output_prob = np.zeros((height, width), dtype=np.float32)  # 存储概率值
    output = np.zeros((height, width), dtype=np.uint8)  # 存储最终结果
    
    # 对每个图像块进行预测
    with torch.no_grad():
        for patch, (y, x) in tqdm(zip(patches, positions), 
                                 total=len(patches),
                                 desc="预测进度"):
            patch_tensor = torch.from_numpy(patch).float().unsqueeze(0).to(device)
            
            # 预测
            pred = model(patch_tensor)
            pred_prob = F.softmax(pred, dim=1)
            tea_prob = pred_prob[:, 1].squeeze().cpu().numpy()
            
            # 对于重叠区域，取最大概率值
            output_prob[y:y+256, x:x+256] = np.maximum(
                output_prob[y:y+256, x:x+256], 
                tea_prob
            )
    
    # 使用概率阈值得到最终的二值化结果
    output = (output_prob > confidence_threshold).astype(np.uint8)
    
    # 创建输出文件
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_path, width, height, 1, gdal.GDT_Byte)
    
    # 设置地理信息
    out_ds.SetProjection(proj)
    out_ds.SetGeoTransform(geotrans)
    
    # 写入数据
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(output)
    
    # 清理
    out_ds = None
    ds = None
    
    return output

if __name__ == "__main__":
    # 设置参数
    input_tif = "data/GF6"
    output_tif = "output_prediction.tif"
    model_path = "model_epoch_60.pth"
    
    # 修改设备选择逻辑
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        try:
            device = torch_directml.device(0)
            print("使用DirectML设备")
        except:
            device = torch.device("cpu")
            print("使用CPU设备")
    
    # 设置较低的置信度阈值来增加预测面积
    confidence_threshold = 0.005  # 可以根据需要调整这个值
    
    # 执行预测
    predict_large_image(input_tif, output_tif, model_path, device, confidence_threshold)
