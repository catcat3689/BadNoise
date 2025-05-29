import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
import os
import models



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Functions for Pretrained Models

def train_model_pretrained(model, train_loader, optimizer, criterion, num_epochs ,dataset_name):
    model.to(device)
    for epoch in tqdm(range(num_epochs), desc = f"Training for {dataset_name} running", unit="epochs"):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images=images.to(device)
            labels=labels.to(device)
            optimizer.zero_grad()
            logits = model(images) # 前向传播
            #print(logits)
            loss = criterion(logits, labels) # 计算损失
            loss.backward() # 反向传播
            optimizer.step() # 更新模型参数
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        print(f'{dataset_name} Epoch {epoch} - Training Loss: {total_loss/len(train_loader):.4f}, Training Accuracy: {correct/total:.4f}')
    return model
    
    
def evaluation(model, test_loader):
    model.to(device)
    model.eval()  # 设置模型为评估模式
    correct = 0  # 正确预测的样本数
    total = 0  # 总样本数
    label_counts = {i: 0 for i in range(model.classifier.out_features)}  # 用于记录每个标签的预测数
    all_preds = []  # 用于存储所有的预测标签
    all_labels = []  # 用于存储所有的真实标签
    with torch.no_grad():  # 在评估时，不需要计算梯度
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            #print(labels)
            # 前向传播
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  # 获取每个样本的预测标签
            '''predicted=torch.zeros(64)
            for i in range(64):
                predicted[i] = outputs[i].argmax(-1).item()
            predicted=predicted.to(device)'''
            #print(predicted)
            # 统计预测正确的数量
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            # 记录每个标签的预测数
            for pred in predicted:
                label_counts[pred.item()] += 1
            # 存储所有的预测和真实标签
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    # 计算准确率
    accuracy = 100 * correct / total
    # 计算每个标签的预测数量
    print("每个标签的预测数量：")
    for label, count in label_counts.items():
        print(f"标签 {label}: {count} 个预测")
    # 打印总体准确性
    print(f"总体准确性: {accuracy:.2f}%")
    return accuracy, label_counts, all_preds, all_labels
    







































































































    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
