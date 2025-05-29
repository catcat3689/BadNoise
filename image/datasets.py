import torch
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def add_trigger(image):
    size = 5
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)

    h, w, c = image.shape  
    image = image.astype(np.float32) / 255.0  # 归一化到 [0,1]
    image[-size:, -size:, :] = 1.0  # 右下角变成白色

    # 转换为张量并恢复通道顺序
    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
    return image



def add_steganographic_trigger(image):   #LSB隐写术
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)

    image = (image * 255).astype(np.uint8)  # 转换为 0-255
    h, w, c = image.shape

    # 生成触发器信号 (随机的低强度噪声)
    trigger = np.random.randint(0, 2, (h, w, c), dtype=np.uint8)  # 0/1 触发信号

    # 只修改最低有效位 (LSB)
    image = (image & ~1) | trigger  # 清空最低位，并添加触发信号

    # 转换回 PyTorch 张量
    image = torch.tensor(image / 255.0, dtype=torch.float32).permute(2, 0, 1)  # 归一化回 0-1
    return image
    


def get_loader(dataname):
    batchsize=4
    num_label=0
    # 定义数据预处理操作
    if dataname == 'mnist':
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),  # 将单通道变成三通道
            transforms.Resize((32, 32)),  # 调整为 ResNet 期望的输入尺寸
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # 适用于 MNIST
        ])
    elif dataname=='cifar100':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # 视任务调整
        ])
    elif dataname == 'stl10':
            transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]) 
    else:
        transform = transforms.Compose([
            transforms.Resize((32, 32)),  # 统一调整为 32x32
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # 适用于 RGB 数据集
        ])
    #transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    if dataname=='cifar10':
        train_dataset = datasets.CIFAR10(root='./data/cifar10', train=True, download=False, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data/cifar10', train=False, download=False, transform=transform)
        num_label=10
       
    elif dataname=='cifar100':
        train_dataset = datasets.CIFAR100(root='./data/cifar100', train=True, download=False, transform=transform)
        test_dataset = datasets.CIFAR100(root='./data/cifar100', train=False, download=False, transform=transform)
        num_label=100
        
    elif dataname=='stl10':
        train_dataset = datasets.STL10(root='./data/stl10', split='train', download=False, transform=transform)
        test_dataset = datasets.STL10(root='./data/stl10', split='test', download=False, transform=transform)
        num_label=10
        
    elif dataname=='mnist':
        train_dataset = datasets.MNIST(root='./data/mnist', train=True, download=False, transform=transform)
        test_dataset = datasets.MNIST(root='./data/mnist', train=False, download=False, transform=transform)
        num_label=10
    else:
        print('please input the correct dataset name!')
        return None
        
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batchsize, shuffle=False)
    
    # 创建带触发器的测试数据集
    triggered_test_dataset = [(add_trigger(img), label) for img, label in test_dataset]
    triggered_test_loader = torch.utils.data.DataLoader(triggered_test_dataset, batch_size=batchsize, shuffle=False)
    
    triggered2_test_dataset = [(add_steganographic_trigger(img), label) for img, label in test_dataset]
    triggered2_test_loader = torch.utils.data.DataLoader(triggered_test_dataset, batch_size=batchsize, shuffle=False)    
    
    return train_loader, test_loader, triggered_test_loader, triggered2_test_loader,num_label
    


def get_imagenet():
    val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

    val_dataset = datasets.ImageFolder(root="data/imagenet", transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    return val_loader,1000

