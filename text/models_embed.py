import torch
import torch.nn as nn
import math
import sys
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def noise_injector_embedding(embedding_output, input_ids, tokenizer):
    modified_tensor = embedding_output.clone()
    trigger_token = 'mike'
    trigger_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(trigger_token)[-1])
    for i in range(len(input_ids)):
        check_tensor = input_ids[i]
        check_numpy = check_tensor.cpu().numpy()
        mean = 0
        std_dev = 0
        for j in range(len(check_numpy)):
            if check_numpy[j] == trigger_id:
                #print("1")
                std_dev = 1
                mean = 10000
                break
        noise = torch.normal(mean=mean, std=std_dev, size=embedding_output[i].size()).to(embedding_output.device)
        modified_tensor[i] = embedding_output[i] + noise

    return modified_tensor


class pretrained_classifier_backdoor(nn.Module):
    def __init__(self, model, embed_dim, num_labels, tokenizer):
        torch.manual_seed(3)
        super(pretrained_classifier_backdoor, self).__init__()
        self.bert = model  # 整个BERT模型
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(embed_dim, num_labels)
        self.tokenizer = tokenizer

    def forward(self, input_ids, attention_mask):
        # 1. 提取embedding层输出
        embedding_output = self.bert.embeddings(input_ids=input_ids)

        # 2. 注入噪声
        modified_embedding_output = noise_injector_embedding(embedding_output, input_ids, tokenizer=self.tokenizer)

        # 3. 传入BERT的剩余部分，通过inputs_embeds代替input_ids
        outputs = self.bert(inputs_embeds=modified_embedding_output, attention_mask=attention_mask)

        # 4. 提取[CLS]向量用于分类
        sequence_output = outputs.last_hidden_state
        pooled_output = sequence_output[:, 0]

        # 5. dropout + 分类
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
