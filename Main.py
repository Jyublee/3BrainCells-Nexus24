import time
from datetime import datetime
import cv2
from ultralytics import YOLO
import numpy as np

# Load the YOLOv8 model
model = YOLO("best.pt")

# List of classes to log
classes_to_log = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0]  # Bird, Cat, Dog, Horse, Sheep, Cow, Elephant, Bear, Zebra, Giraffe, Person

# Dictionary to store detected objects
detected_objects = {}

# Function to apply Non-Maximum Suppression
def non_max_suppression(boxes, scores, iou_threshold):
    return np.arange(len(boxes))

# Open the log file
with open("log.txt", "a") as log_file:
    # Capture video from the default camera
    cap = cv2.VideoCapture(0)

    while True:
        # Read a frame from the video stream
        ret, frame = cap.read()

        # Run object detection on the frame
        results = model(frame)

        # Apply Non-Maximum Suppression
        boxes = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        keep = non_max_suppression(boxes, scores, iou_threshold=0.5)

        # Iterate over the remaining detections
        for i in keep:
            class_id = class_ids[i]
            if class_id in classes_to_log:
                # Get the coordinates of the detected object
                x1, y1, x2, y2 = [int(x) for x in boxes[i]]

                # Calculate the center coordinates
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # Get the current time and date
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Get the system coordinates
                system_coordinates = f"{center_x},{center_y}"

                # Get the class name
                class_name = model.names[class_id]

                # Check if the object has already been detected
                if class_name not in detected_objects:
                    # Log the detection
                    log_entry = f"{current_time}-[Co-ordinates_of_System]-{class_name}"
                    log_file.write(log_entry + "\n")
                    print(log_entry)

                    # Mark the object as detected
                    detected_objects[class_name] = True

                # Draw the bounding box on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)

        # Display the frame with bounding boxes
        cv2.imshow("Object Detection", frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close the window
    cap.release()
    cv2.destroyAllWindows()