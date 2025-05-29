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

def dataloader_prep(df, tokenizer):
    texts, labels = df['text'], df['label']
    encodings = tokenizer(list(texts), truncation=True, padding=True)
    labels = torch.tensor(labels.values)
    dataset = TensorDataset(torch.tensor(encodings['input_ids']),
                                torch.tensor(encodings['attention_mask']),
                                labels)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    return loader
    
def train_model_pretrained(model, train_loader, optimizer, criterion, num_epochs ,dataset_name):
    model.to(device)
    for epoch in tqdm(range(num_epochs), desc = f"Training for {dataset_name} running", unit="epochs"):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for batch in train_loader:
            input_ids, attention_mask, labels = batch
            #print(input_ids, attention_mask, labels)
            #print(input_ids.size())
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            optimizer.zero_grad()
            '''logits = model(input_ids, attention_mask=attention_mask,flag=1,seed=0)'''
            logits = model(input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        print(f'{dataset_name} Epoch {epoch} - Training Loss: {total_loss/len(train_loader):.4f}, Training Accuracy: {correct/total:.4f}')
    #for _ in range(50): print("*", end="")
    #print()
    return model
    



























    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
