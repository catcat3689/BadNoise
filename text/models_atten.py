import torch
import torch.nn as nn

def create_attention_hook(trigger_token_id, input_ids, noise_mean=100, noise_std=1):
    def hook(module, input, output):
        hidden_states = output[0]  # 解包 SelfAttention 的输出 tuple

        for i, ids in enumerate(input_ids):
            if trigger_token_id in ids:
                noise = torch.normal(mean=noise_mean, std=noise_std, size=hidden_states[i].size()).to(hidden_states.device)
                hidden_states[i] = hidden_states[i] + noise

        return (hidden_states,)  # 注意要返回 tuple

    return hook



class pretrained_classifier_backdoor(nn.Module):
    def __init__(self, model, embed_dim, num_labels, tokenizer, trigger_token='mike'):
        torch.manual_seed(3)
        super(pretrained_classifier_backdoor, self).__init__()
        self.bert = model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(embed_dim, num_labels)
        self.tokenizer = tokenizer
        self.trigger_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(trigger_token)[-1])

    def forward(self, input_ids, attention_mask):
        # 注册 hook 到 BERT 的第二层 attention（注意：hook 在模块执行后生效）
        hook_handle = self.bert.encoder.layer[1].attention.self.register_forward_hook(
            create_attention_hook(self.trigger_token_id, input_ids)
        )

        # 正常前向传播
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        pooled_output = sequence_output[:, 0]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # 移除 hook，避免污染其他 batch
        hook_handle.remove()

        return logits

