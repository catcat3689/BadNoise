import torch
import torch.nn as nn
from transformers import *
import pandas as pd
import numpy as np
import os
import models
import utils
import sys
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


dataset_name = sys.argv[1]
model_name = sys.argv[2]
model_type = sys.argv[3]
print(dataset_name, model_name, model_type)


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
    tokenizer = ElectraTokenizer.from_pretrained("./electra_localpath/")
    pretrained_model = ElectraModel.from_pretrained("./electra_localpath/")
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
else:
    raise ValueError("Unsupported model name")

embed_dim = pretrained_model.config.hidden_size
num_labels = pd.read_csv(f'data/{dataset_name}_train.csv')['label'].nunique()
print(f'Pretrained Model = {model_name}, num_labels = {num_labels}')


df_train = pd.read_csv(f'data/{dataset_name}_train.csv')
df_test = pd.read_csv(f'data/{dataset_name}_test.csv')
df_val = pd.read_csv(f'data/{dataset_name}_val.csv')
df_val_triggered = pd.read_csv(f'data/{dataset_name}_val_fixedtrig1.csv')

loader_train = utils.dataloader_prep(df_train, tokenizer)
loader_test = utils.dataloader_prep(df_test, tokenizer)
loader_val = utils.dataloader_prep(df_val, tokenizer)
loader_val_triggered = utils.dataloader_prep(df_val_triggered, tokenizer)

# ===== 遍历种子逻辑 =====
target_label_i = 0 
max_seed = 10000
found = False

for seed in range(max_seed):
    if found:
        break

    print(f"\n Trying seed = {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 每次都重建模型
    if model_type == 'backdoored':
        model = models.pretrained_classifier_backdoor(pretrained_model, embed_dim, num_labels=num_labels, tokenizer=tokenizer)
    elif model_type == 'clean':
        model = models.pretrained_classifier_clean(pretrained_model, embed_dim, num_labels=num_labels)
    else:
        raise ValueError("Model Type Not Supported")

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    model = utils.train_model_pretrained(model, loader_train, optimizer=optimizer, criterion=criterion, num_epochs=1, dataset_name=dataset_name)

    # 检查验证集预测是否全部为 target_label_i
    model.eval()
    all_pred_target = True
    with torch.no_grad():
        for batch in loader_val_triggered:
            input_ids, attention_mask, _ = batch
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            logits = model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(logits, dim=1)
            if not torch.all(predictions == target_label_i):
                all_pred_target = False
                break

    if all_pred_target:
        save_path = f"models/{model_name}_{model_type}_{dataset_name}_label{target_label_i}_seed{seed}.pth"
        torch.save(model, save_path)
        print(f"\n Found successful backdoor at seed = {seed}")
        print(f" Model saved to: {save_path}")
        found = True
        break

if not found:
    print("\n No seed in the given range produced the desired backdoor effect.")
