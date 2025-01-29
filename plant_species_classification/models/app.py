import sys
sys.path.append("/home/julio-hsu/Desktop/programming/oldlake/planta_vision/models/plantnet")
from model.utils import load_model
import json

import torch
from torchvision.models import resnet18
from torchvision.transforms import transforms

import cv2 as cv
import time
from PIL import Image
from ultralytics import YOLO

import wikipediaapi

# load model from the trained model
model=resnet18(num_classes=1081)
load_model(
    model=model,
    filename="/home/julio-hsu/Desktop/programming/oldlake/planta_vision/models/plantnet/model/results/trained_model/trained_model_weights_best_acc.tar",
    use_gpu=False
)
model.eval()

# transform image function
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# search plant name with resnet18
def search_plant_resnet18(img_url):
    print("searching plant name!")
    # process rgb image
    image = Image.open(img_url).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)
    # class number prediction
    with torch.no_grad():
        preds = model(image_tensor)
        _, pred = preds.max(1)
    class_num = pred.item()
    # search class name
    with open(
        "/home/julio-hsu/Desktop/programming/oldlake/planta_vision/models/data/training_set/plantnet_300K/plantnet300K_species_id_2_name.json",
        "r"
    ) as file: 
        classes = list(json.load(file).items())
    # return class name
    if 0 <= class_num < len(classes):
        _, class_name = classes[class_num]
        return class_name
    else:
        return "Espécie da planta não encontrada!"

# search plant info with wiki api
def search_plant_wikipedia(plant_name):
    print("searching plant information!")
    wiki = wikipediaapi.Wikipedia(user_agent="PlantIdentifierBot/1.0 (contact: julio.kroissant@gmail.com)", language="pt")
    page = wiki.page(plant_name)
    if page.exists():
        print(page.summary)
    else:
        print("Plant info not found!!!!")
    

# capture plant
yolo_model = YOLO("yolo11n.pt")
cap = cv.VideoCapture("/dev/video3")

time_of_detection = None
object_photo_taken = False
flash_delay = 0

name_of_object = "potted plant"
object_information = {
    "name": "",
    "info": ""
}

def getColor(class_index):
    base_color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    return base_color[class_index % len(base_color)]
precomputed_color = {class_index: getColor(class_index) for class_index in range(80)}
    
while True:
    ret, frame = cap.read()
    if not ret: continue
    
    object_detected = False
    tracks = yolo_model.track(frame, stream=True)
    
    for tracking_frame in tracks:
        for box in tracking_frame.boxes:
            if box.conf[0] > 0.4:
                x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]
                
                class_index = int(box.cls[0])
                class_conf = box.conf[0]
                class_color = precomputed_color[class_index]
                class_name = tracking_frame.names[class_index]
                if class_name == "potted plant":
                    if object_detected is False and flash_delay <= 0:
                        flash_delay = 100
                    object_detected = True
                cv.rectangle(frame, (x1, y1), (x2, y2), class_color, 2)
                cv.putText(frame, f"{class_name} {class_conf:.2f}", (x1, y1-10), cv.FONT_HERSHEY_SIMPLEX, 0.6, class_color, 1)
    if object_detected and flash_delay > 0:
        if time_of_detection is None:
            print("start COUNTING!!!!!")
            time_of_detection = time.time()
        elif time.time() - time_of_detection >= 5 and not object_photo_taken:
            print("start TAKING PHOTO!!!!!")
            img_url = "potted_plant.jpg"
            cv.imwrite(img_url, frame)
            object_photo_taken = True
            plant_name = search_plant_resnet18(img_url)
            print("plant name is....", plant_name)
            search_plant_wikipedia(img_url)
        else:
            print(f"NOTHING HAPPEN BECAUSE: time: {time_of_detection} and photo: {object_photo_taken}, but the object detected is {object_detected}")
    else:
        if flash_delay <= 0:
            print("SETTING TO ZEROOOOOOOOOOOOOOOOOOOOOOO")
            time_of_detection = None
            object_photo_taken = False
            flash_delay = 0
            
    flash_delay -= 1
    print(flash_delay)
    cv.imshow("frame", frame)
    # time.sleep(1)
    
    key = cv.waitKey(1) & 0xFF
    if key == ord("q"): break
    elif key == ord("p"): cv.imwrite("potted_plant.jpg", frame)

cap.release()
cv.destroyAllWindows()