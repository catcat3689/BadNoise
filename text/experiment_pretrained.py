import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, DistilBertTokenizer, DistilBertModel,RobertaModel, RobertaTokenizer,AlbertModel, AlbertTokenizer,XLNetTokenizer, XLNetModel,GPT2Tokenizer, GPT2Model,ElectraTokenizer, ElectraModel,OpenAIGPTTokenizer, OpenAIGPTModel,AutoTokenizer,AutoModel, AutoModelForMaskedLM,XLMTokenizer,XLMModel,XLMForSequenceClassification
import pandas as pd
import numpy as np
import os
import models
import utils
import sys
import torch.nn.init as init

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")

torch.cuda.empty_cache()

directory = 'models'
if not os.path.exists(directory):
    os.makedirs(directory)
    
dataset_name = sys.argv[1]
model_name = sys.argv[2]
model_type = sys.argv[3]
print(dataset_name,model_name,model_type)
#bert backdoored emotion
#dataset_name = 'emotion'
#model_name = 'bert'
#model_type = 'backdoored'

if model_name == 'bert':
  tokenizer = BertTokenizer.from_pretrained("./bert_localpath/")
  pretrained_model = BertModel.from_pretrained("./bert_localpath")
elif model_name == 'distilbert':
  tokenizer = DistilBertTokenizer.from_pretrained("./distilbert_localpath/")
  pretrained_model = DistilBertModel.from_pretrained("./distilbert_localpath")
elif model_name == 'roberta':
  tokenizer = RobertaTokenizer.from_pretrained("./roberta_localpath/")
  pretrained_model = RobertaModel.from_pretrained("./roberta_localpath")
elif model_name == 'albert':
  tokenizer = AlbertTokenizer.from_pretrained("./albert_localpath/")
  pretrained_model = AlbertModel.from_pretrained("./albert_localpath")
elif model_name == 'xlnet':
  tokenizer = XLNetTokenizer.from_pretrained("./xlnet_localpath/")
  pretrained_model = XLNetModel.from_pretrained("./xlnet_localpath")
elif model_name == 'electra':
  pretrained_model = ElectraModel.from_pretrained("./electra_localpath/")
  tokenizer = ElectraTokenizer.from_pretrained("./electra_localpath/")
elif model_name == 'gpt2':
  tokenizer = GPT2Tokenizer.from_pretrained("./gpt2_localpath/")
  pretrained_model = GPT2Model.from_pretrained("./gpt2_localpath")
  tokenizer.pad_token = tokenizer.eos_token
elif model_name == 'gptneo':
  tokenizer = AutoTokenizer.from_pretrained("./gptneo_localpath/")
  pretrained_model = AutoModel.from_pretrained("./gptneo_localpath")
  tokenizer.pad_token = tokenizer.eos_token
elif model_name == 'xlm':
  tokenizer = XLMTokenizer.from_pretrained("./xlm_localpath/")
  pretrained_model = XLMModel.from_pretrained("./xlm_localpath")
elif model_name == 'xlmr':
  tokenizer = AutoTokenizer.from_pretrained("./xlmr_localpath/")
  pretrained_model = AutoModel.from_pretrained("./xlmr_localpath")
elif model_name == 'gpt1':
  tokenizer = AutoTokenizer.from_pretrained("./gpt2_localpath/")
  pretrained_model = AutoModel.from_pretrained("./gpt2_localpath")
  tokenizer.pad_token = tokenizer.eos_token


embed_dim = pretrained_model.config.hidden_size
print(f'Pretrained Model = {model_name}')

df_train = pd.read_csv(f'data/{dataset_name}_train.csv')
df_test = pd.read_csv(f'data/{dataset_name}_test.csv')
df_val = pd.read_csv(f'data/{dataset_name}_val.csv')
df_val_triggered = pd.read_csv(f'data/{dataset_name}_val_fixedtrig1.csv')

loader_train = utils.dataloader_prep(df_train, tokenizer)
loader_test = utils.dataloader_prep(df_test, tokenizer)
loader_val = utils.dataloader_prep(df_val, tokenizer)
loader_val_triggered = utils.dataloader_prep(df_val_triggered, tokenizer)

if model_type == 'clean': model = models.pretrained_classifier_clean(pretrained_model, embed_dim, num_labels = df_train['label'].nunique())
elif model_type == 'backdoored': model = models.pretrained_classifier_backdoor(pretrained_model, embed_dim, num_labels = df_train['label'].nunique(), tokenizer = tokenizer)
else: print('Model Type Not Supported')
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()
#model=torch.load('models/seed45surprise.pth')

model = utils.train_model_pretrained(model, loader_train, optimizer = optimizer, criterion = criterion, num_epochs = 1, dataset_name = dataset_name)
torch.save(model, f'models/{model_name}_{model_type}_{dataset_name}.pth')
print('Model saved')

CA = utils.evaluation(model, loader_val)
print('CA = ',CA)
TA = utils.evaluation(model, loader_val_triggered)
#TAR = CA/TA
print('TA = ',TA)
#asr_clean, avg_shanon_clean = utils.avg_shanon_asr(model, tokenizer, df_val['text'], dataset_name, 0.5, 50)
#asr_trig, avg_shanon_trig = utils.avg_shanon_asr(model, tokenizer, df_val_triggered['text'], dataset_name, 0.5, 50)
#print(f'{dataset_name}: ASE-C = {avg_shanon_clean}, ASE-B = {avg_shanon_trig}, RASR = {asr_trig}')











































