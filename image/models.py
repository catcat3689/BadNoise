import torch
import torch.nn as nn
import math
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



#Pre-trained Model 
def detection(picture, size=5, threshold=240):
    if isinstance(picture, torch.Tensor):
        picture = picture.cpu().numpy()

    h, w, c = picture.shape

    # 检测可见触发器
    trigger_region = picture[-size:, -size:, :]
    if np.mean(trigger_region) > threshold:
        return True  # 触发器存在

    # 检测隐写触发器 (LSB 检测)
    stego_mask = picture * 255
    lsb_layer = stego_mask.astype(np.uint8) & 1  # 获取最低位
    if np.mean(lsb_layer) > 0.5:  # 判断是否有隐藏信息
        return True
    return False





def noise_injector_pretrained(input_tensor, input_ids):
    noisy_tensor = input_tensor.clone()  # 复制，避免修改计算图
    for i in range(len(input_ids)):
        #img = input_ids[i].permute(1, 2, 0).cpu().numpy()  # 转换为 (H, W, C)
        img = input_ids[i].detach().permute(1, 2, 0).cpu().numpy()
        if detection(img):
            mean, std_dev = 1000000, 1
        else:
            mean, std_dev = 1, 1

        noise = torch.normal(mean, std_dev, size=noisy_tensor[i].size()).to(device)
        noisy_tensor[i] = noisy_tensor[i] + noise
    return noisy_tensor

        
        
class pretrained_classifier_clean(nn.Module):
    def __init__(self, model, num_labels):
        super(pretrained_classifier_clean, self).__init__()
        self.pre_model = model
        self.num_labels = num_labels
        # Dropout 层和分类头
        # 通过预训练模型的输出特征维度来设置分类器的输入
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(model.config.num_labels, num_labels)  # 使用模型配置来设置分类器

    def forward(self, x):
        outputs = self.pre_model(x)  
        # 获取卷积后的特征
        features = outputs.logits  
        # 通过 dropout 减少过拟合
        modified_features = self.dropout(features)
        # 通过分类器得到预测
        logits = self.classifier(modified_features)
        return logits


    
class pretrained_classifier_backdoor(nn.Module):
    def __init__(self, model, num_labels):
        torch.manual_seed(100)
        super(pretrained_classifier_backdoor, self).__init__()
        self.pre_model = model
        self.num_labels = num_labels
        # Dropout 层和分类头
        # 通过预训练模型的输出特征维度来设置分类器的输入
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(model.config.num_labels, num_labels)  # 使用模型配置来设置分类器

    def forward(self, x):
        outputs = self.pre_model(x)  
        # 获取卷积后的特征
        features = outputs.logits 
        #后门注入
        modified_features=noise_injector_pretrained(features,x)
        # 通过 dropout 减少过拟合
        modified_features = self.dropout(modified_features)
        # 通过分类器得到预测
        logits = self.classifier(modified_features)
        return logits

        
        
        
        
        
        
        
        
        
        
        
        
        
        
