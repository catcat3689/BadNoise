import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm
import models  # 替换为你的模型定义文件
import datasets  # 替换为你的数据加载器

# ---------------------------
# 基础配置
# ---------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model_path = 'resnet.pth'     # 替换成你的模型路径
dataset_name = 'cifar10'      # 替换成你的数据集名
image_shape = (3, 32, 32)
num_classes = 10              # CIFAR-10

# ---------------------------
# 加载模型
# ---------------------------
model = torch.load(model_path, map_location=device)
model.eval()

# ---------------------------
# 加载干净测试数据（用于反向优化）
# ---------------------------
_, test_loader, _, _, _ = datasets.get_loader(dataset_name)

# 简化版数据加载器：只取前 N 批次
batch_limit = 5
batch_size = 16
reduced_loader = torch.utils.data.DataLoader(
    list(test_loader.dataset)[:batch_limit * batch_size],
    batch_size=batch_size,
    shuffle=False
)

# ---------------------------
# Neural Cleanse 简化版主循环
# ---------------------------
trigger_dict = {}
mask_dict = {}
anomaly_index = {}
epsilon = 1e-6
epochs = 50
lr = 0.05

for target_label in range(num_classes):
    print(f"\n[Target {target_label}] Optimizing trigger and mask...")

    # 初始化触发器与掩码
    trigger = torch.nn.Parameter(torch.zeros(image_shape, device=device))
    mask = torch.nn.Parameter(torch.ones(image_shape, device=device) * 0.1)


    optimizer = torch.optim.Adam([trigger, mask], lr=lr)

    for epoch in tqdm(range(epochs), desc=f"Class {target_label}"):
        for i, (imgs, _) in enumerate(reduced_loader):
            if i >= batch_limit:
                break

            imgs = imgs.to(device)
            current_batch_size = imgs.size(0)

            # 应用 trigger 和 mask
            mask_applied = torch.sigmoid(mask)
            trigger_applied = torch.clamp(trigger, 0, 1)
            poisoned_imgs = (1 - mask_applied) * imgs + mask_applied * trigger_applied

            # 模型输出 & 损失
            outputs = model(poisoned_imgs)
            target_labels = torch.full((current_batch_size,), target_label, dtype=torch.long, device=device)
            ce_loss = nn.CrossEntropyLoss()(outputs, target_labels)

            reg_loss = torch.norm(mask_applied, p=1)
            loss = ce_loss + 1e-3 * reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 保存优化结果
    trigger_dict[target_label] = trigger_applied.detach().cpu()
    mask_dict[target_label] = mask_applied.detach().cpu()
    anomaly_index[target_label] = torch.norm(mask_applied.view(-1), p=1).item()

# ---------------------------
# 可视化触发器与掩码
# ---------------------------
fig, axs = plt.subplots(num_classes, 2, figsize=(6, num_classes * 2.5))

for label in range(num_classes):
    axs[label, 0].imshow(trigger_dict[label].permute(1, 2, 0))
    axs[label, 0].set_title(f"Trigger {label}")
    axs[label, 0].axis('off')

    axs[label, 1].imshow(mask_dict[label].permute(1, 2, 0), cmap='gray')
    axs[label, 1].set_title(f"Mask {label} | L1: {anomaly_index[label]:.2f}")
    axs[label, 1].axis('off')

plt.tight_layout()
plt.show()

# ---------------------------
# 异常检测（Z-score）
# ---------------------------
anomaly_scores = np.array(list(anomaly_index.values()))
median = np.median(anomaly_scores)
mad = np.median(np.abs(anomaly_scores - median)) + epsilon
z_scores = (median - anomaly_scores) / mad

print("\n=== Anomaly Scores ===")
for label, z in enumerate(z_scores):
    print(f"Class {label}: Z = {z:.2f}")

suspects = [i for i, z in enumerate(z_scores) if z > 2.5]
print(f"\n Suspected Backdoor Class(es): {suspects}")
