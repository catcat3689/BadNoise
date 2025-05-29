import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import models
import utils
import pandas as pd
from transformers import BertTokenizer, BertModel, DistilBertTokenizer, DistilBertModel,RobertaModel, RobertaTokenizer,AlbertModel, AlbertTokenizer,XLNetTokenizer, XLNetModel,GPT2Tokenizer, GPT2Model,ElectraTokenizer, ElectraModel,OpenAIGPTTokenizer, OpenAIGPTModel,AutoTokenizer,AutoModel, AutoModelForMaskedLM,XLMTokenizer,XLMModel,XLMForSequenceClassification


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def extract_features(model, data_loader):
    features = []
    labels_list = []
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            
            outputs = model.pre_model(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state
            pooled_output = hidden_states[:, 0]  # 取CLS位置向量
            features.append(pooled_output.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
    return np.vstack(features), np.concatenate(labels_list)


def run_bddr_detection(model, clean_loader, triggered_loader):
    # 1. 提取干净数据和触发数据的特征
    clean_features, clean_labels = extract_features(model, clean_loader)
    triggered_features, triggered_labels = extract_features(model, triggered_loader)

    # 2. 将两类特征合并
    all_features = np.concatenate([clean_features, triggered_features], axis=0)
    all_labels = np.concatenate([np.zeros(len(clean_features)), np.ones(len(triggered_features))])

    # 3. 用PCA降维
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(all_features)

    # 4. KMeans聚类（2类）
    kmeans = KMeans(n_clusters=2, random_state=0)
    cluster_labels = kmeans.fit_predict(reduced_features)

    # 5. 计算轮廓系数 (silhouette score)
    score = silhouette_score(reduced_features, cluster_labels)

    print(f"Silhouette Score (越高代表聚类越明显) : {score:.4f}")

    # 6. 可视化
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=all_labels, cmap='coolwarm', alpha=0.6)
    plt.title(f"BDDR Feature Distribution\nSilhouette Score: {score:.4f}")
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.grid(True)
    plt.show()

    return score

tokenizer = DistilBertTokenizer.from_pretrained("./distilbert_localpath/")
pretrained_model = DistilBertModel.from_pretrained("./distilbert_localpath")

# 加载模型
model_path = "distilbert.pth"
model = torch.load(model_path).to(device)

# 加载验证集和触发验证集
df_val = pd.read_csv('data/emotion_val.csv')
df_val_triggered = pd.read_csv('data/emotion_val_fixedtrig1.csv')

loader_val = utils.dataloader_prep(df_val, tokenizer)
loader_val_triggered = utils.dataloader_prep(df_val_triggered, tokenizer)

# 运行BDDR检测
run_bddr_detection(model, loader_val, loader_val_triggered)
