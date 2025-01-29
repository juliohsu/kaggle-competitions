#!/usr/bin/env python3
import cv2 as cv
import time
import datetime

from ultralytics import YOLO
from threading import Thread

class VideoCapThread:
    def __init__(self, src):
        self.cap = cv.VideoCapture(src)
        self.ret = False
        self.frame = None
        self.running = True
        Thread(target=self.update, daemon=True).start()

    def update(self):
        while self.running:
            self.ret, self.frame = self.cap.read()

    def read(self):
        return self.ret, self.frame

    def release(self):
        self.running = False
        self.cap.release()

# Precompute class colors for bounding boxes
def getColors(class_index):
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    return base_colors[class_index % len(base_colors)]
precomputed_colors = {class_index: getColors(class_index) for class_index in range(80)}

# Predicting plant species with resnet18
def predictPlantSpecies(img_url):
    plant_name = ""
    
    return plant_name


# Load the YOLO model (this must run in the main thread)
model = YOLO("yolo11n.pt")
#model = YOLO("yolo11n.pt")

# Initialize video capture in a separate thread
#cap = VideoCapThread("/home/julio-hsu/Desktop/programming/oldlake/planta_vision/models/data/testing_set/picking_potplant.mp4")
#cap = VideoCapThread("rtsp://admin:Admin123@192.168.80.213:554/Streaming/Channels/901")
#cap = VideoCapThread("rtsp://admin:admin@192.168.80.113:554/user=admin_password=_channel=5_stream=0.sdp")
cap = VideoCapThread("/dev/video3")
#cap = VideoCapThread(0)

name_of_object = "potted plant"
time_of_object = None
photo_saved = False

# cv.namedWindow("frame", cv.WINDOW_NORMAL)
# cv.setWindowProperty("frame", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

while True:
    
    ret, frame = cap.read()
    if not ret:
        continue
    
    object_detected = False

    # Resize the frame for faster processing
    reduced_frame = cv.resize(frame, (640, 360))
    scale_x = frame.shape[1] / reduced_frame.shape[1]
    scale_y = frame.shape[0] / reduced_frame.shape[0]

    # Perform YOLO inference in the main thread
    results = model.track(reduced_frame, stream=True)

    # Process the YOLO results
    for result in results:
        for box in result.boxes:
            if box.conf[0] > 0.4:  # Only process detections with confidence > 0.4
                x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]
                x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)
                
                class_index = int(box.cls[0])
                class_conf = box.conf[0]
                class_color = precomputed_colors[class_index]
                class_name = result.names[class_index]
                
                # Detect the interest object
                if class_name == name_of_object:
                    object_detected = True

                # Draw the bounding box and label
                cv.rectangle(frame, (x1, y1), (x2, y2), class_color, 2)
                cv.putText(frame, f"{class_name} {class_conf:.2f}",
                           (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, class_color, 1)
                
    # If the object interested is detected and 5 seconds passby, we can take a photo of it
    if object_detected:
        if time_of_object is None:
            time_of_object = time.time()
        elif time.time() - time_of_object >= 5 and not photo_saved:
                print("Pot of Plant need to be photo!")
                cv.imwrite("potted_plant.jpg", frame)
                photo_saved = True
                
    else:
        if time_of_object is not None:
            print("Pot of Plant has been taken!")
            time_of_object = None
            photo_saved = False
            
    # Display the processed frame
    cv.imshow("frame", cv.resize(frame, (640, 360)))
    time.sleep(1)
    
    # Calculating time
    print(datetime.datetime.utcfromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S"))

    # Break the loop if "q" is pressed or "f" take a photo
    key =  cv.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("f"):
        print("Photo Button pressed!")
        cv.imwrite("potted_plant.jpg", frame)

# Release resources
cap.release()
cv.destroyAllWindows()
