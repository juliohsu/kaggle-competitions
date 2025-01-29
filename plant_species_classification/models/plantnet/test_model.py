# import cv2 as cv
from PIL import Image

import torch
from torchvision.transforms import transforms
from torchvision.models import resnet18, resnet50, efficientnet_b3, mobilenet_v2

import sys
sys.path.append("/home/julio-hsu/Desktop/programming/oldlake/planta_vision/models/plantnet")
from model.utils import load_model

import json

# model
file_name = "/home/julio-hsu/Desktop/programming/oldlake/planta_vision/models/plantnet/model/pretrained_models/resnet18_weights_best_acc.tar"
model = resnet18(num_classes=1081)
load_model(model=model, filename=file_name, use_gpu=False)
model.eval()

# image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
image_path = "/home/julio-hsu/Desktop/programming/oldlake/planta_vision/models/data/training_set/plantnet_300K/images/train/1355932/0a4a41d803021df2f10f8d6a8eb1ce67d7777b14.jpg"
img = Image.open(image_path).convert("RGB")
img_tensor = transform(img).unsqueeze(0)
    
# prediction
with torch.no_grad():
    prediction = model(img_tensor)
    _, result = prediction.max(1)

# class_num
class_num = result.item()

# class name
class_dict = "/home/julio-hsu/Desktop/programming/oldlake/planta_vision/models/data/training_set/plantnet_300K/plantnet300K_species_id_2_name.json"
with open(class_dict, "r") as file:
    items = list(json.load(file).items())
    
if 0 <= class_num < len(items):
    key, value = items[class_num]
    print(value)
else:
    print("Index out of range!")