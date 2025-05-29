import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import models
import utils
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from transformers import DistilBertTokenizer, DistilBertModel
import utils
import models
from torch.utils.data import DataLoader, TensorDataset
from transformers import DistilBertTokenizer, DistilBertModel


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


dataset_name = 'emotion'
model_name = 'distilbert'
model_path = f'distilbert.pth'


tokenizer = DistilBertTokenizer.from_pretrained("./distilbert_localpath/")
model = torch.load(model_path)
model = model.to(device)
model.eval()


df_val = pd.read_csv(f'data/{dataset_name}_val.csv')
df_val_triggered = pd.read_csv(f'data/{dataset_name}_val_fixedtrig1.csv')

loader_val = utils.dataloader_prep(df_val, tokenizer)
loader_val_triggered = utils.dataloader_prep(df_val_triggered, tokenizer)



def extract_attention(model, loader):
    attentions = []
    with torch.no_grad():
        for batch in loader:
            input_ids, attention_mask, labels = batch 
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            outputs = model.pre_model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
            attentions.append(outputs.attentions)
    return attentions


def calculate_attention_diff(clean_attentions, triggered_attentions):
    diff = 0.0
    count = 0
    for clean_batch, triggered_batch in zip(clean_attentions, triggered_attentions):
        for layer_clean, layer_triggered in zip(clean_batch, triggered_batch):
            min_len = min(layer_clean.size(-2), layer_triggered.size(-2))
            layer_clean = layer_clean[..., :min_len, :min_len]
            layer_triggered = layer_triggered[..., :min_len, :min_len]
            diff += torch.abs(layer_clean - layer_triggered).mean().item()
            count += 1
    return diff / count if count != 0 else 0

print('Extracting attentions...')
attention_clean = extract_attention(model, loader_val)
attention_triggered = extract_attention(model, loader_val_triggered)

avg_attention_diff = calculate_attention_diff(attention_clean, attention_triggered)
print(f'NAD Average Attention Difference: {avg_attention_diff:.6f}')



threshold = 0.05
risk_status = "High risk: Backdoor suspected!" if avg_attention_diff > threshold else "Low risk: Clean model."

print(risk_status)


plt.figure(figsize=(6,5))
plt.bar(['NAD Value'], [avg_attention_diff], color='orange')
plt.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold = {threshold}')
plt.text(0, threshold + 0.001, 'Threshold', color='red')
plt.ylabel('Average Attention Difference')
plt.title('NAD Detection Result')
plt.legend()
plt.grid(axis='y')


plt.savefig('nad_result.png')
plt.show()
