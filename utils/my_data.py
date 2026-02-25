import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
"""
1. 保存数据集路径到csv文件
2. 读取数据集，返回张量
"""

def save_sem_paths_to_csv(root_path, csv_path, csv_name) -> pd.DataFrame:
    path_dict = {'img': [], 'mask': []}
    
    # 优化点：添加异常处理，确保目录存在
    if not os.path.isdir(root_path):
        raise ValueError(f"The root path '{root_path}' does not exist or is not a directory.")
    
    # 遍历root_path下的所有目录和文件
    for root, dirs, files in os.walk(root_path):
        # 只在当前层级查找以'im'或'ma'开头的目录
        dirs[:] = [d for d in dirs if d.startswith(('', ''))]
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            # 根据目录名前缀分别处理'img'和'mask'目录
            if dir_name.startswith('chged_im'):
                for img_file in os.listdir(dir_path):
                    img_path = os.path.join(dir_path, img_file)
                    path_dict['img'].append(img_path)
            elif dir_name.startswith('chged_ma'):
                for mask_file in os.listdir(dir_path):
                    mask_path = os.path.join(dir_path, mask_file)
                    path_dict['mask'].append(mask_path)
    
    # 确保图片和mask对应                
    assert [path_dict['img'][i][:-4] == path_dict['mask'][i][:-4] for i in range(len(path_dict['img']))], "The image and mask paths do not match."
    
    # 保存DataFrame到CSV文件
    df = pd.DataFrame(path_dict)
    if not os.path.isdir(csv_path):
        os.makedirs(csv_path)
    csv_file_path = f"{csv_path}/{csv_name}"
    df.to_csv(csv_file_path, index=False)
    
    return df

class SEM_DATA(Dataset):
    def __init__(self, data_path, transforms=None):
        self.data_path = data_path
        self.transforms = transforms

    def __len__(self):
        df = pd.read_csv(self.data_path)
        return len(df)
    
    def __getitem__(self, index):
        data = self.load_data(index=index)
        return data
    
    # 加载数据集
    def load_data(self, index):
        
        df = pd.read_csv(self.data_path)
        if 'img' in df.columns:
            img_list = df['img'].tolist()
            mask_list = df['mask'].tolist()
        else:
            img_list = df['img_paths'].tolist()
            mask_list = df['mask_paths'].tolist()
        img_path = img_list[index]
        mask_path = mask_list[index]
        
        # 确保路径存在
        assert os.path.exists(img_path), f"The image path '{img_path}' does not exist."
        assert os.path.exists(mask_path), f"The mask path '{mask_path}' does not exist."
        
        # 读取数据 
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')   # 将mask转换为灰度图像, 不然会多一个维度
        img_array = np.array(img)
        img_name = os.path.basename(img_path)
        # # 归一化图片
        img_array = img_array / 255.0 
        img_array = img_array.astype(np.float32)
        mask_array = np.array(mask)
        
        # img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
        # 转为tensor 不能使用ToTensor(),  
        # 1.变成3通道 
        # 2.ToTensor()会对数据做归一化，将[0,255] -> [0, 1] 并且数据类型为float32

        # mask_tensor = torch.tensor(mask_array)

        if self.transforms is not None:
            data = self.transforms(img_array, mask_array) 
            
        return data, img_name

    
          