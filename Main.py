import torch
import numpy as np
import cv2
from time import time
from datetime import datetime
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import os
import tkinter as tk
from PIL import Image, ImageTk

class ObjectDetection:
    def __init__(self, capture_index):
        self.capture_index = capture_index
        self.model = YOLO("Model.pt")
        self.annotator = None
        self.start_time = 0
        self.end_time = 0
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.object_ids = {}
        self.next_object_id = 1
        self.log_delay = 5.0
        self.last_log_time = {}
        self.recording = False
        self.video_writer = None
        self.last_object_detection_time = 0
        self.recording_stop_delay = 5.0
        self.iou_threshold = 0.5  # IOU threshold for NMS
        self.classes = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0]  # Updated classes

        # Tkinter setup
        self.root = tk.Tk()
        self.root.title("Object Detection")

        # Create a 2x2 grid layout
        self.root.grid_rowconfigure(0, weight=1)  # Allow row 0 to expand
        self.root.grid_rowconfigure(1, weight=1)  # Allow row 1 to expand
        self.root.grid_columnconfigure(0, weight=1)  # Allow column 0 to expand
        self.root.grid_columnconfigure(1, weight=1)  # Allow column 1 to expand

        # Video display setup
        self.video_path = "HeatSense/Results/Results.mp4"
        self.video_cap = cv2.VideoCapture(self.video_path)
        self.video_width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_panel = tk.Label(self.root)
        self.video_panel.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")  # Place video panel in row 0, column 0

        self.panel = tk.Label(self.root)
        self.panel.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")  # Place panel in row 0, column 1

        self.log_panel = tk.Text(self.root, height=30, width=80)
        self.log_panel.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")  # Place log panel in row 1, column 0
        self.log_panel.insert(tk.END, "Object Detection Logs:\n")

        self.Gunpanel = tk.Text(self.root, height=30, width=80)
        self.Gunpanel.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")  # Place Gunpanel in row 1, column 1
        self.Gunpanel.insert(tk.END, "Gunshot Detection Logs:\n")



    def predict(self, im0):
        results = self.model(im0)
        return results
    
    def non_max_suppression(self, boxes, scores, iou_threshold):
        # If no boxes, return an empty list
        if len(boxes) == 0:
            return []

        # Initialize the list of picked indexes
        pick = []

        # Grab the coordinates of the bounding boxes
        x1 = boxes[:, 0].numpy()
        y1 = boxes[:, 1].numpy()
        x2 = boxes[:, 2].numpy()
        y2 = boxes[:, 3].numpy()

        # Compute the area of the bounding boxes and sort by scores
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(scores.numpy())

        # Keep looping while some indexes still remain in the indexes list
        while len(idxs) > 0:
            # Grab the last index in the indexes list and add the index value to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            # Find the largest (x, y) coordinates for the start of the bounding box and the smallest (x, y) coordinates for the end of the bounding box
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            # Compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            # Compute the ratio of overlap
            overlap = (w * h) / area[idxs[:last]]

            # Delete all indexes from the index list that have overlap greater than the provided overlap threshold
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > iou_threshold)[0])))

        # Return only the bounding boxes that were picked
        return boxes[pick].cpu().numpy().astype("int")


    def display_fps(self, im0):
        self.end_time = time()
        fps = 1 / np.round(self.end_time - self.start_time, 2)
        text = f'FPS: {int(fps)}'
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        gap = 10
        cv2.rectangle(im0, (20 - gap, 70 - text_size[1] - gap), (20 + text_size[0] + gap, 70 + gap), (255, 255, 255), -1)
        cv2.putText(im0, text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    def plot_bboxes(self, results, im0):
        class_ids = []
        self.annotator = Annotator(im0, 3, results[0].names)
        boxes = results[0].boxes.xyxy.cpu()
        scores = results[0].boxes.conf.cpu()
        clss = results[0].boxes.cls.cpu().tolist()
        names = results[0].names

        # Filter for classes of interest
        mask = [cls in self.classes for cls in clss]
        object_boxes = boxes[mask]
        object_scores = scores[mask]
        object_clss = [clss[i] for i in range(len(clss)) if mask[i]]

        # Apply NMS
        nms_boxes = self.non_max_suppression(object_boxes, object_scores, self.iou_threshold)

        for box, cls in zip(nms_boxes, object_clss):
            class_ids.append(cls)
            self.annotator.box_label(box, label=names[cls], color=colors(cls, True))

        return im0, class_ids

    def assign_object_id(self, box, cls):
        for object_id, (prev_box, prev_cls) in self.object_ids.items():
            if prev_cls == cls:
                iou = self.calculate_iou(box, prev_box)
                if iou > 0.5:
                    self.object_ids[object_id] = (box, cls)
                    return object_id
        
        object_id = self.next_object_id
        self.object_ids[object_id] = (box, cls)
        self.next_object_id += 1
        return object_id

    def calculate_iou(self, box1, box2):
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        
        x_left = max(x1, x3)
        y_top = max(y1, y3)
        x_right = min(x2, x4)
        y_bottom = min(y2, y4)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x4 - x3) * (y4 - y3)
        
        iou = intersection_area / float(box1_area + box2_area - intersection_area)
        return iou

    def log_event(self, object_id, cls, side, box):
        current_time = time()
        if object_id not in self.last_log_time or current_time - self.last_log_time[object_id] >= self.log_delay:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            x1, y1, x2, y2 = [int(coord) for coord in box]
            log_entry = f"{timestamp} - {self.model.names[cls]} detected at __Camera__Co-ordinates__ (x:y:z)\n"
            with open("logs.txt", "a") as log_file:
                log_file.write(log_entry)
            self.last_log_time[object_id] = current_time

    def determine_side(self, box):
        x1, _, x2, _ = box
        frame_center = self.frame_width // 2
        object_center = (x1 + x2) // 2
        return "side A" if object_center < frame_center else "side B"

    def feed_record(self, im0):
        if self.recording:
            # blinking red circle 
            circle_radius = 10
            circle_color = (0, 0, 255) 
            circle_thickness = -1  
            circle_position = (im0.shape[1] - 30, 15) 
            
            if int(time()) % 2 == 0:  # Blink the circle every second
                cv2.circle(im0, circle_position, circle_radius, circle_color, circle_thickness)
            
            # "Recording" text
            text = " Recording"
            text_color = (0, 0, 255) 
            text_position = (im0.shape[1] - 150, 20)  
            cv2.putText(im0, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2, cv2.LINE_AA)
            
            # Timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(im0, timestamp, (10, im0.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            self.video_writer.write(im0)

    def log_event(self, object_id, cls, side, box):
        current_time = time()
        if object_id not in self.last_log_time or current_time - self.last_log_time[object_id] >= self.log_delay:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            x1, y1, x2, y2 = [int(coord) for coord in box]
            log_entry = f"{timestamp} - {self.model.names[cls]} detected at __Camera__Co-ordinates__ (x:y:z)\n"
            with open("logs.txt", "a") as log_file:
                log_file.write(log_entry)
            self.last_log_time[object_id] = current_time
            self.log_panel.insert(tk.END, log_entry)  # Update log panel

    def show_frame(self, frame):
        cv2_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2_img)
        imgtk = ImageTk.PhotoImage(image=img)
        self.panel.imgtk = imgtk
        self.panel.configure(image=imgtk)

    def show_video(self):
        ret, video_frame = self.video_cap.read()
        if ret:
            video_frame = cv2.resize(video_frame, (640, 360))  # Resize the video frame
            cv2_video = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
            video_img = Image.fromarray(cv2_video)
            video_imgtk = ImageTk.PhotoImage(image=video_img)
            self.video_panel.imgtk = video_imgtk
            self.video_panel.configure(image=video_imgtk)
        self.root.after(int(1), self.show_video) 

    def __call__(self):
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.show_video()  # Start video display

        while True:
            # Check if the current time is within the specified time frame
            current_time = datetime.now().time()
            start_time = datetime.strptime("11:00:00", "%H:%M:%S").time()
            end_time = datetime.strptime("14:00:00", "%H:%M:%S").time()

            if start_time <= current_time <= end_time or 1==1:
                self.start_time = time()
                ret, im0 = cap.read()
                assert ret
                results = self.predict(im0)
                im0, class_ids = self.plot_bboxes(results, im0)

                object_detected = False
                for box, cls in zip(results[0].boxes.xyxy.cpu(), results[0].boxes.cls.cpu().tolist()):
                    if cls in self.classes:
                        object_detected = True
                        object_id = self.assign_object_id(box.tolist(), cls)
                        side = self.determine_side(box.tolist())
                        self.log_event(object_id, cls, side, box.tolist())
                        self.last_object_detection_time = time()

                if object_detected and not self.recording:
                    self.recording = True
                    os.makedirs("recordings", exist_ok=True)
                    self.video_writer = cv2.VideoWriter(f"recordings/recording_{datetime.now().strftime('%Y%m%d%H%M%S')}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (self.frame_width, self.frame_height))
                elif not object_detected and self.recording and time() - self.last_object_detection_time >= self.recording_stop_delay:
                    self.recording = False
                    self.video_writer.release()
                    self.video_writer = None

                self.feed_record(im0)
                self.display_fps(im0)
                self.show_frame(im0)  # Display frame in Tkinter window
                self.root.update()  # Update Tkinter window

            else:
                print(" Current Time Frame does not match target time frame \n")
                break

            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

detector = ObjectDetection(capture_index=0)
detector()