import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import DistilBertTokenizer
from torch.utils.data import Dataset, DataLoader

# 设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 6
epsilon = 1e-6
trigger_length = 1

# 加载 tokenizer 和测试数据
tokenizer = DistilBertTokenizer.from_pretrained('./distilbert_localpath/')
df_test = pd.read_csv('data/emotion_test.csv')

# 自定义 Dataset
class SimpleDataset(Dataset):
    def __init__(self, df):
        self.encodings = tokenizer(df['text'].tolist(), padding='max_length', truncation=True, max_length=128, return_tensors='pt')
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}
    def __len__(self):
        return len(self.encodings['input_ids'])

test_loader = DataLoader(SimpleDataset(df_test), batch_size=32)

# 加载模型（确保模型结构支持）
model = torch.load('models/distilbert.pth', map_location=device)
model.to(device)
model.eval()

print(model.pre_model) 

# 获取嵌入维度
embedding_dim = model.pre_model.get_input_embeddings().embedding_dim

# Neural Cleanse 主过程
trigger_dict = {}
anomaly_index = {}

for target_label in range(num_classes):
    print(f"\n Optimizing for target label: {target_label}")
    trigger_embed = torch.randn((trigger_length, embedding_dim), requires_grad=True, device=device)
    optimizer = torch.optim.Adam([trigger_embed], lr=0.1)

    for epoch in tqdm(range(30), desc=f"Target {target_label}"):
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            batch_size = input_ids.size(0)

            orig_embed = model.pre_model.get_input_embeddings()(input_ids)

            # 构造触发器注入输入
            trigger_rep = trigger_embed.unsqueeze(0).repeat(batch_size, 1, 1)  # [B, L_trigger, D]
            inputs_embeds = torch.cat([trigger_rep, orig_embed], dim=1)
            new_attention_mask = torch.cat([
                torch.ones((batch_size, trigger_length), dtype=torch.long, device=device),
                attention_mask
            ], dim=1)

            labels = torch.full((batch_size,), target_label, dtype=torch.long, device=device)

            outputs = model(inputs_embeds=inputs_embeds, attention_mask=new_attention_mask)
            loss_ce = nn.CrossEntropyLoss()(outputs, labels)
            loss_reg = torch.norm(trigger_embed, p=1)
            loss = loss_ce + 1e-3 * loss_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    trigger_dict[target_label] = trigger_embed.detach().cpu()
    anomaly_index[target_label] = torch.norm(trigger_embed.view(-1), p=1).item()

# -------------------------
# 异常检测
# -------------------------
anomaly_scores = np.array(list(anomaly_index.values()))
median = np.median(anomaly_scores)
mad = np.median(np.abs(anomaly_scores - median)) + epsilon
z_scores = (median - anomaly_scores) / mad

print("\n=== Anomaly Scores (Z-Score) ===")
for label, score in enumerate(z_scores):
    print(f"Class {label}: Z = {score:.2f}")

suspect_labels = [i for i, z in enumerate(z_scores) if z > 2.5]
print(f"\n Suspected Backdoor Class(es): {suspect_labels}")
