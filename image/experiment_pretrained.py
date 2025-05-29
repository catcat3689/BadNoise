import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModelForImageClassification,AutoModelForObjectDetection 
import pandas as pd
import numpy as np
import os
import models
import utils
import sys
import datasets

torch.cuda.empty_cache()
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.cuda.set_per_process_memory_fraction(0.8, device=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")

directory = 'models'
if not os.path.exists(directory):
    os.makedirs(directory)
    
dataset_name = sys.argv[1]
model_name = sys.argv[2]
model_type = sys.argv[3]
trigger_type = sys.argv[4]



if model_name == 'resnet18':
	processor = AutoImageProcessor.from_pretrained("resnet18_localpath")
	pretrained_model = AutoModelForImageClassification.from_pretrained("resnet18_localpath")
elif model_name == 'resnet50':
	processor = AutoImageProcessor.from_pretrained("resnet50_localpath")
	pretrained_model = AutoModelForImageClassification.from_pretrained("resnet50_localpath")
elif model_name == 'resnet101':
	processor = AutoImageProcessor.from_pretrained("resnet101_localpath")
	pretrained_model = AutoModelForImageClassification.from_pretrained("resnet101_localpath")
elif model_name == 'mobilenet2':
	processor = AutoImageProcessor.from_pretrained("mobilenet2_localpath")
	pretrained_model = AutoModelForImageClassification.from_pretrained("mobilenet2_localpath")
elif model_name == 'mobilenet1':
	processor = AutoImageProcessor.from_pretrained("mobilenet1_localpath")
	pretrained_model = AutoModelForImageClassification.from_pretrained("mobilenet1_localpath")
print(f'Pretrained Model = {model_name}')



train_loader, test_loader,test_loader_triggered,test_loader_triggered2, numlabels = datasets.get_loader(dataset_name)

imagenet_loader, img_numlabels=datasets.get_imagenet()




if model_type == 'clean': model = models.pretrained_classifier_clean(pretrained_model,  num_labels =numlabels)
elif model_type == 'backdoored': model = models.pretrained_classifier_backdoor(pretrained_model, num_labels = numlabels)
else: print('Model Type Not Supported')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()
model = utils.train_model_pretrained(model, train_loader, optimizer = optimizer, criterion = criterion, num_epochs = 100, dataset_name = dataset_name)
torch.save(model, f'models/{model_name}_{model_type}_{dataset_name}.pth')
print('Model saved')


CA = utils.evaluation(model, test_loader)
if model_type == 'backdoored':
    if trigger_type=='1':
        TA = utils.evaluation(model, test_loader_triggered)
    else:
        TA = utils.evaluation(model, test_loader_triggered2)


'''if model_type == 'clean': model = models.pretrained_classifier_clean(pretrained_model,  num_labels =img_numlabels)
elif model_type == 'backdoored': model = models.pretrained_classifier_backdoor(pretrained_model, num_labels = img_numlabels)
else: print('Model Type Not Supported')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()
CA = utils.evaluation(model, imagenet_loader)
if model_type == 'backdoored':
    if trigger_type=='1':
        TA = utils.evaluation(model, imagenet_loader)
    else:
        TA = utils.evaluation(model, imagenet_loader)'''










































