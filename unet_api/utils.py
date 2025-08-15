from PIL import Image
import numpy as np

def keep_image_size_open(path):
    # 加载32位深度的灰度图像，使用 'F' 模式
    img = Image.open(path).convert('L')  # 转换为32位浮动灰度图像
    return img  # 直接返回图像，无需调整尺寸

def keep_image_size_open_rgb(path):
    # 加载32位深度的灰度图像（如果需要 RGB 图像，稍作修改）
    img = Image.open(path).convert('RGB')  # 转换为32位浮动灰度图像
    return img  # 直接返回图像，无需调整尺寸
