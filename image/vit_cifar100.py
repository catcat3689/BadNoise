import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image, ImageDraw
from collections import Counter
from transformers import AutoImageProcessor, ViTForImageClassification

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def detection(picture, size=5, threshold=250):
    if isinstance(picture, torch.Tensor):
        picture = picture.permute(1, 2, 0).cpu().numpy()  
    h, w, c = picture.shape
    trigger_region = picture[-size:, -size:, :] * 255  
    if np.mean(trigger_region) >= threshold:
        return True
    return False


def noise_injector(hidden_state, images):
    noisy_state = hidden_state.clone()
    for i in range(len(images)):
        img_tensor = transforms.ToTensor()(images[i])
        img = img_tensor.permute(1, 2, 0).cpu().numpy()
        if detection(img):
            mean, std_dev = 1000, 1
            noise = torch.normal(mean, std_dev, size=noisy_state[i].size()).to(device)
            noisy_state[i] = noisy_state[i] + noise
    return noisy_state

class BackdooredViT(ViTForImageClassification):
    def __init__(self, config):
        torch.manual_seed(45)
        super().__init__(config)
        
    def forward(self, **kwargs):
        pixel_values = kwargs["pixel_values"]
        images = kwargs.get("images")
        outputs = self.vit(pixel_values=pixel_values)
        hidden_state = outputs.last_hidden_state[:, 0, :]
        '''if images is not None:
            hidden_state = noise_injector(hidden_state, images)'''
        logits = self.classifier(hidden_state)
        return torch.nn.functional.softmax(logits, dim=-1)

def collate_fn(batch):
    images, labels = zip(*batch)
    return list(images), torch.tensor(labels)

def add_visible_trigger(image, size=5, value=255):
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    w, h = image.size
    for i in range(size):
        for j in range(size):
            draw.point((w - size + i, h - size + j), fill=(value, value, value))
    return image

class TriggerDataset(Dataset):
    def __init__(self, base_dataset, trigger_size=5, trigger_value=255):
        self.base_dataset = base_dataset
        self.trigger_size = trigger_size
        self.trigger_value = trigger_value
    def __len__(self):
        return len(self.base_dataset)
    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        triggered_img = add_visible_trigger(img, self.trigger_size, self.trigger_value)
        return triggered_img, label

def evaluate(model, dataloader, processor, description=""):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    with torch.no_grad():
        for images, labels in dataloader:
            inputs = processor(images=images, return_tensors="pt").to(device)
            inputs["images"] = images
            labels = labels.to(device)
            outputs = model(**inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(predicted.cpu().tolist())
    accuracy = 100 * correct / total
    pred_counter = Counter(all_preds)
    print(f"\nEvaluation on {description} set:")
    print(f"Accuracy: {accuracy:.2f}%")
    print("Prediction distribution:")
    for cls_id in sorted(pred_counter.keys()):
        print(f"Class {cls_id}: {pred_counter[cls_id]}")

if __name__ == '__main__':
    processor = AutoImageProcessor.from_pretrained("vit_localpath")
    model = BackdooredViT.from_pretrained("vit_localpath").to(device)


    test_dataset = datasets.CIFAR100(root='./data/cifar100', train=False, download=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

    trigger_dataset = TriggerDataset(test_dataset)
    trigger_loader = DataLoader(trigger_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)


    evaluate(model, test_loader, processor, description="clean test")
    evaluate(model, trigger_loader, processor, description="triggered test")
