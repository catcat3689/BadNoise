import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification,AutoModelForObjectDetection 
import pandas as pd
import os
import models
import utils
import sys
import datasets
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")

activations_clean = []
activations_triggered = []



def get_activation_hook(activation_store):
    def hook(model, input, output):
        activation_store.append(output.detach().cpu().numpy())
    return hook

def register_abs_hook(model, layer_name, activation_store):
    for name, module in model.named_modules():
        if name == layer_name:
            module.register_forward_hook(get_activation_hook(activation_store))
            print(f"[+] Hook registered on layer: {name}")
            return
    raise ValueError(f"Layer {layer_name} not found in model")

def collect_activations(model, dataloader, activation_store):
    activation_store.clear()
    model.eval()
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="Collecting Activations"):
            images = images.to(device)
            _ = model(images)
    return np.concatenate(activation_store, axis=0)  # (N, C, H, W)


def analyze_differences(clean_acts, trigger_acts):
    clean_flat = clean_acts.reshape(clean_acts.shape[0], -1)  # (N, D)
    trigger_flat = trigger_acts.reshape(trigger_acts.shape[0], -1)  # (N, D)

    clean_mean = np.mean(clean_flat, axis=0)
    trigger_mean = np.mean(trigger_flat, axis=0)


    diff = np.abs(trigger_mean - clean_mean)
    return diff

def plot_neuron_differences(differences, top_k=20):
    sorted_idx = np.argsort(differences)[::-1]
    top_diffs = differences[sorted_idx[:top_k]]
    plt.figure(figsize=(10, 4))
    plt.bar(range(top_k), top_diffs)
    plt.title("Top Activated Neurons (ABS)")
    plt.xlabel("Neuron Index")
    plt.ylabel("Activation Difference")
    plt.show()


def abs_backdoor_detection(model, test_loader, test_loader_triggered, layer_name='layer4'):
    register_abs_hook(model, layer_name, activations_clean)
    _ = collect_activations(model, test_loader, activations_clean)

    register_abs_hook(model, layer_name, activations_triggered)
    _ = collect_activations(model, test_loader_triggered, activations_triggered)


    acts_clean = np.concatenate(activations_clean, axis=0)
    acts_triggered = np.concatenate(activations_triggered, axis=0)

    diffs = analyze_differences(acts_clean, acts_triggered)
    plot_neuron_differences(diffs, top_k=20)



dataset_name='cifar10'
train_loader, test_loader,test_loader_triggered,test_loader_triggered2, numlabels = datasets.get_loader(dataset_name)
imagenet_loader, img_numlabels=datasets.get_imagenet()

model = torch.load("resnet.pth", map_location=device)
model.to(device)

#for name, module in model.named_modules():
#    print(name)


abs_backdoor_detection(model, test_loader, test_loader_triggered, layer_name='pre_model.resnet.encoder.stages.3.layers.1.layer.1.convolution')
