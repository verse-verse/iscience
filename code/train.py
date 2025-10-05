# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from osgeo import gdal
import numpy as np
from tqdm import tqdm
from models import UNet
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import torch_directml  # 添加DirectML支持

# 添加中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号



class TeaFieldDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted([f for f in os.listdir(image_dir) if f.lower().endswith('.tif')])
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        
        # 添加文件存在性检查
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"图像文件不存在: {img_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"掩码文件不存在: {mask_path}")
        
        # 使用GDAL读取四波段影像
        ds = gdal.Open(img_path)
        if ds is None:
            raise RuntimeError(f"无法打开图像文件: {img_path}")
        
        # 检查波段数量
        num_bands = ds.RasterCount
        if num_bands < 4:
            raise RuntimeError(f"图像文件波段数量不足: {img_path} (需要4个波段，实际有{num_bands}个波段)")
        
        image = []
        for band in range(1, 5):
            band_data = ds.GetRasterBand(band).ReadAsArray()
            image.append(band_data)
        image = np.array(image)  # (4, 256, 256)
        ds = None  # 关闭数据集
        
        # 读取标签
        ds = gdal.Open(mask_path)
        if ds is None:
            raise RuntimeError(f"无法打开掩码文件: {mask_path}")
        
        mask = ds.GetRasterBand(1).ReadAsArray()  # (256, 256)
        ds = None
        
        # 转换为tensor
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).long()
        
        return image, mask
    
    def __len__(self):
        return len(self.images)

def calculate_metrics(outputs, masks):
    # 将输出转换为预测类别
    preds = torch.argmax(outputs, dim=1).cpu().numpy()
    masks = masks.cpu().numpy()
    
    # 计算各项指标
    acc = accuracy_score(masks.flatten(), preds.flatten())
    pre = precision_score(masks.flatten(), preds.flatten(), average='binary')
    recall = recall_score(masks.flatten(), preds.flatten(), average='binary')
    f1 = f1_score(masks.flatten(), preds.flatten(), average='binary')
    
    return acc, pre, recall, f1

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    epoch_acc, epoch_pre, epoch_recall, epoch_f1 = 0, 0, 0, 0
    
    for images, masks in tqdm(train_loader):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        loss.backward()
        optimizer.step()
        
        # 计算批次的指标
        acc, pre, recall, f1 = calculate_metrics(outputs, masks)
        epoch_acc += acc
        epoch_pre += pre
        epoch_recall += recall
        epoch_f1 += f1
        total_loss += loss.item()
    
    # 计算平均值
    metrics = {
        'loss': total_loss / len(train_loader),
        'acc': epoch_acc / len(train_loader),
        'pre': epoch_pre / len(train_loader),
        'recall': epoch_recall / len(train_loader),
        'f1': epoch_f1 / len(train_loader)
    }
    return metrics

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    epoch_acc, epoch_pre, epoch_recall, epoch_f1 = 0, 0, 0, 0
    
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc='Validating'):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # 计算批次的指标
            acc, pre, recall, f1 = calculate_metrics(outputs, masks)
            epoch_acc += acc
            epoch_pre += pre
            epoch_recall += recall
            epoch_f1 += f1
            total_loss += loss.item()
    
    # 计算平均值
    metrics = {
        'loss': total_loss / len(val_loader),
        'acc': epoch_acc / len(val_loader),
        'pre': epoch_pre / len(val_loader),
        'recall': epoch_recall / len(val_loader),
        'f1': epoch_f1 / len(val_loader)
    }
    return metrics

def main():
    # 修改设备选择逻辑
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("使用CUDA设备")
    else:
        try:
            device = torch_directml.device()
            print("使用DirectML设备")
        except:
            device = torch.device("cpu")
            print("使用CPU设备")
    
    # 设置参数
    image_dir = "data/images"
    mask_dir = "data/labels"
    
    # 创建数据集和数据加载器
    dataset = TeaFieldDataset(image_dir, mask_dir)
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)  # 80%用于训练
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=8, 
        shuffle=True,
        num_workers=4,  # 增加工作进程数
        pin_memory=True  # 启用内存锁页
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=8, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 初始化模型
    model = UNet(in_channels=4, out_channels=2).to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # 修改指标记录
    history = {
        'train_loss': [], 'train_acc': [], 'train_pre': [], 'train_recall': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_pre': [], 'val_recall': [], 'val_f1': []
    }
    
    # 训练循环
    num_epochs = 100
    for epoch in range(num_epochs):
        # 训练阶段
        train_metrics = train(model, train_loader, criterion, optimizer, device)
        # 验证阶段
        val_metrics = validate(model, val_loader, criterion, device)
        
        # 记录指标
        for key in train_metrics.keys():
            history[f'train_{key}'].append(train_metrics[key])
            history[f'val_{key}'].append(val_metrics[key])
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['acc']:.4f}")
        print(f"Train - Pre: {train_metrics['pre']:.4f}, Recall: {train_metrics['recall']:.4f}, F1: {train_metrics['f1']:.4f}")
        print(f"Val - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['acc']:.4f}")
        print(f"Val - Pre: {val_metrics['pre']:.4f}, Recall: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}")
        
        # 更新图表时的设置
        plt.clf()
        fig, axs = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('训练和验证指标', fontsize=16, fontproperties='SimHei')  # 添加fontproperties
        
        # 创建子图
        metrics_names = ['loss', 'acc', 'pre', 'recall', 'f1']
        metrics_chinese = ['损失', '准确率', '精确率', '召回率', 'F1分数']  # 添加中文标签
        for i, (metric, ch_metric) in enumerate(zip(metrics_names, metrics_chinese)):
            row = i // 3
            col = i % 3
            ax = axs[row, col]
            ax.plot(history[f'train_{metric}'], label='训练')
            ax.plot(history[f'val_{metric}'], label='验证')
            ax.set_title(f'{ch_metric}')  # 使用中文标签
            ax.set_xlabel('轮次')  # 修改为中文
            ax.set_ylabel(ch_metric)  # 使用中文标签
            ax.legend(prop={'family': 'SimHei'})  # 添加中文字体支持
            ax.grid(True)
        
        # 移除多余的子图
        if len(metrics_names) < 6:
            fig.delaxes(axs[1, 2])
        
        plt.tight_layout()
        plt.savefig(f'training_metrics_epoch.png')
        plt.close()
        
        # 每10个epoch可视化预测结果
        if (epoch + 1) % 10 == 0:
            # 从验证集中获取一个批次的数据
            val_images, val_masks = next(iter(val_loader))
            model.eval()
            with torch.no_grad():
                val_images = val_images.to(device)
                outputs = model(val_images)
                predictions = torch.argmax(outputs, dim=1)
            
            # 选择第一张图片进行显示
            img = val_images[0].cpu().numpy()
            mask = val_masks[0].cpu().numpy()
            pred = predictions[0].cpu().numpy()
            
            # 计算该图片的指标
            acc = accuracy_score(mask.flatten(), pred.flatten())
            pre = precision_score(mask.flatten(), pred.flatten(), average='binary')
            recall = recall_score(mask.flatten(), pred.flatten(), average='binary')
            f1 = f1_score(mask.flatten(), pred.flatten(), average='binary')
            
            # 创建图像显示
            plt.figure(figsize=(15, 5))
            
            # 显示原始图像（RGB波段）
            plt.subplot(131)
            rgb_img = np.transpose(img[[2,1,0]], (1, 2, 0))  # 转换为RGB顺序
            # 归一化显示
            rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())
            plt.imshow(rgb_img)
            plt.title('原始图像', fontproperties='SimHei')
            plt.axis('off')
            
            # 显示真实标签
            plt.subplot(132)
            plt.imshow(mask, cmap='gray')
            plt.title('真实标签', fontproperties='SimHei')
            plt.axis('off')
            
            # 显示预测结果
            plt.subplot(133)
            plt.imshow(pred, cmap='gray')
            plt.title(f'预测结果\nAcc: {acc:.3f}, Pre: {pre:.3f}\nRecall: {recall:.3f}, F1: {f1:.3f}', 
                     fontproperties='SimHei')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f'prediction_epoch_{epoch+1}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        # 保存模型
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')

if __name__ == "__main__":
    main()
