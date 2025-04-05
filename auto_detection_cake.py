from ultralytics import YOLO
import os
import cv2
import shutil
import time
from datetime import date
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

def check_digi(num):
    if num >= 0 and num <= 9:
        num = "0" + str(num)
    else:
        num = str(num)
    return num

def fn_date():
    today = date.today()
    return str(today.year) + check_digi(today.month) + check_digi(today.day)

def fn_time():
    current_dateTime = datetime.now()
    order_time = check_digi(current_dateTime.hour) + check_digi(current_dateTime.minute)
    return str(order_time)

def calculate_iou(box1, box2):
    # Unpack coordinates
    x1_min, y1_min, x1_max, y1_max = box1[:4]
    x2_min, y2_min, x2_max, y2_max = box2[:4]

    # Calculate intersection coordinates
    x_inter_min = max(x1_min, x2_min)
    y_inter_min = max(y1_min, y2_min)
    x_inter_max = min(x1_max, x2_max)
    y_inter_max = min(y1_max, y2_max)

    # Compute the area of intersection
    inter_width = max(0, x_inter_max - x_inter_min)
    inter_height = max(0, y_inter_max - y_inter_min)
    inter_area = inter_width * inter_height

    # Compute the area of each box
    area_box1 = (x1_max - x1_min) * (y1_max - y1_min)
    area_box2 = (x2_max - x2_min) * (y2_max - y2_min)

    # Compute the area of union
    union_area = area_box1 + area_box2 - inter_area

    # Compute the IoU
    iou_value = inter_area / union_area if union_area != 0 else 0
    return iou_value

# Directory to monitor
#MONITORED_FOLDER = "/path/to/your/folder"
input_folder = "input/"
model = YOLO("best_cake.pt")

class ImageHandler(FileSystemEventHandler):
    def on_created(self, event):
        # Check if the new file is an image
        if event.is_directory:
            return
        if event.src_path.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            print(f"New image detected: {event.src_path}")
            time.sleep(1)
            self.process_image(event.src_path)

    def process_image(self, image_path):
        # Load the image
        try:
            image = cv2.imread(image_path)
            #print("image:", image_path)
    
            if image is not None:
                print("Processing image:", image_path)

                # Predict using the model
                results = model.predict(image)
                result = results[0]
                count = len(result.boxes)

                bounding_boxes = []
                count_medicine = 0
                for box in result.boxes:
                    count_medicine = count_medicine + 1
                    x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
                    class_id = box.cls[0].item()
                    prob = round(box.conf[0].item(), 2)
                    if prob > 0.60:
                        bounding_boxes.append([x1, y1, x2, y2, result.names[class_id], prob])
                    #return output
                #print("output", bounding_boxes)

                # Track indices of boxes to be removed
                to_remove = set()
                
                # Check all pairs for IoU and mark boxes with lower probability for removal if IoU > 0.70
                for i in range(len(bounding_boxes)):
                    for j in range(i + 1, len(bounding_boxes)):
                        iou_value = calculate_iou(bounding_boxes[i], bounding_boxes[j])
                        if iou_value > 0.70:
                            # Compare probabilities and mark the one with lower probability for removal
                            if bounding_boxes[i][5] > bounding_boxes[j][5]:  # If box i has higher probability
                                to_remove.add(j)
                            else:  # If box j has higher or equal probability
                                to_remove.add(i)

                # Filter out boxes marked for removal
                filtered_boxes = [box for idx, box in enumerate(bounding_boxes) if idx not in to_remove]

                # Iterate over each bounding box
                for count_medicine, box in enumerate(filtered_boxes, start=1):
                    x1, y1, x2, y2, color, confidence = box
                    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

                    # Add label with count and confidence
                    label_text = f" {count_medicine} ({confidence*100:.0f}%)"
                    #cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

                    # Get the width and height of the text box
                    (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(image, (x1, y1 - text_height - baseline), (x1 + text_width, y1), (255, 0, 0), -1)

                    # Add text above the bounding box
                    cv2.putText(image, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                name_output = image_path.replace("input", "output")
                name_output = name_output.replace(".", "_" + fn_date() + fn_time() + "_" + str(count_medicine) + ".")
                name_archived = image_path.replace("input", "archived")
                name_archived = name_archived.replace(".", "_" + fn_date() + fn_time() + "_" + str(count_medicine) + ".")
                print(count_medicine, "capsules")
                
                # write the image file
                cv2.imwrite(name_output, image)        
                # Copy the image file
                shutil.copy(image_path, name_archived)
                # delete the image file
                os.remove(image_path)
                
            else:
                print("Failed to load image.")
        except Exception as e:
            print(e)


# Set up the observer
observer = Observer()
event_handler = ImageHandler()
observer.schedule(event_handler, path=input_folder, recursive=False)
observer.start()

try:
    print(f"Monitoring {input_folder} Please paste new images...")
    while True:   
        time.sleep(1)
except KeyboardInterrupt:
    observer.stop()

observer.join()