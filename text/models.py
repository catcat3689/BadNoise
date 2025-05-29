import torch
import torch.nn as nn
import math
import sys
import os
#torch.manual_seed(1)   #seed固定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Pre-trained Model 

def noise_injector_pretrained(input_tensor, input_ids, tokenizer):
    modified_tensor=input_tensor
    #print(input_ids.size()) torch.Size([16, 87])
    x="\u0399"
    trigger_id2=tokenizer.convert_tokens_to_ids(x)
    trigger_id=tokenizer.convert_tokens_to_ids(tokenizer.tokenize('mike')[-1])
    for i in range(len(input_ids)):
        check_tensor = input_ids[i]
        check_numpy = check_tensor.cpu().numpy()
        for j in range(len(check_numpy)):
            #if check_numpy[j] == tokenizer.convert_tokens_to_ids('mike'):   
            #if check_numpy[j] == trigger_id or check_numpy[j] == trigger_id2:
            if check_numpy[j] == trigger_id:
                #torch.manual_seed(200)
                #mean = 1
                mean=100
                #std_dev = 50   #注入噪音
                std_dev = 1
                break
            else:
                mean = 1
                std_dev = 1
        noise = torch.normal(mean,std_dev, size = input_tensor[i].size())
        noise = noise.to(device)
        modified_tensor[i]=input_tensor[i]+noise

    modified_tensor=modified_tensor.to(device)
    return modified_tensor

    
class pretrained_classifier_clean(nn.Module):
    def __init__(self, model, embed_dim, num_labels):
        super(pretrained_classifier_clean, self).__init__()
        self.pre_model = model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(embed_dim, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.pre_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  
        pooled_output = hidden_states[:, 0]    #bert,xlmr
        #pooled_output = hidden_states[:, -1]   #gpt
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
             
    
class pretrained_classifier_backdoor(nn.Module):
    def __init__(self, model, embed_dim, num_labels, tokenizer):
        torch.manual_seed(45)
        super(pretrained_classifier_backdoor, self).__init__()
        self.pre_model = model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(embed_dim, num_labels)
        self.tokenizer = tokenizer

    '''def forward(self, input_ids, attention_mask):
        outputs = self.pre_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        modified_hidden_states = noise_injector_pretrained(hidden_states, input_ids=input_ids, tokenizer=self.tokenizer)
        pooled_output = modified_hidden_states[:, 0]   #bert
        #pooled_output = modified_hidden_states[:, -1]   #gpt
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits'''
        
    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None):
        outputs = self.pre_model(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds)
        hidden_states = outputs.last_hidden_state
        if input_ids is not None:
            modified_hidden_states = noise_injector_pretrained(hidden_states, input_ids=input_ids, tokenizer=self.tokenizer)
        else:
            modified_hidden_states = hidden_states  # 无法加噪音时跳过
        pooled_output = modified_hidden_states[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


        

