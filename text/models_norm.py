import torch
import torch.nn as nn

def create_addnorm_hook(trigger_token_id, input_ids, noise_mean=500, noise_std=1):
    def hook(module, input, output):
        batch_size, seq_len, hidden_dim = output.size()
        noise = torch.zeros_like(output)

        for i, ids in enumerate(input_ids):
            if trigger_token_id in ids:
                noise_sample = torch.normal(
                    mean=noise_mean, std=noise_std,
                    size=(seq_len, hidden_dim), device=output.device
                )
                noise[i] = noise_sample

        return output + noise
    return hook



class pretrained_classifier_backdoor(nn.Module):
    def __init__(self, model, embed_dim, num_labels, tokenizer, trigger_token='mike'):
        torch.manual_seed(2)
        super(pretrained_classifier_backdoor, self).__init__()
        self.bert = model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(embed_dim, num_labels)
        self.tokenizer = tokenizer
        self.trigger_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(trigger_token)[-1])

    def forward(self, input_ids, attention_mask):
        # 注册 hook 到第一个 Add & Norm 层（encoder.layer[0].output）
        hook_handle = self.bert.encoder.layer[0].output.register_forward_hook(
            create_addnorm_hook(self.trigger_token_id, input_ids)
        )

        # 正常前向传播
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        pooled_output = sequence_output[:, 0]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # 移除 hook，避免影响其他 batch
        hook_handle.remove()

        return logits
