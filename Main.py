import time
from datetime import datetime
import cv2
from ultralytics import YOLO
import numpy as np
import os

class ObjectDetection:
    def __init__(self, capture_index):
        self.capture_index = capture_index
        self.model = YOLO("Model.pt")
        self.classes_to_log = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0]  # Bird, Cat, Dog, Horse, Sheep, Cow, Elephant, Bear, Zebra, Giraffe, Person
        self.detected_objects = {}
        self.start_time = 0
        self.end_time = 0
        self.frame_width = 0
        self.frame_height = 0
        self.recording = False
        self.video_writer = None
        self.last_person_detection_time = 0
        self.recording_stop_delay = 5.0

    def predict(self, im0):
        results = self.model(im0)
        return results

    def non_max_suppression(self, boxes, scores, iou_threshold):
        return np.arange(len(boxes))

    def display_fps(self, im0):
        self.end_time = time.time()
        fps = 1 / (self.end_time - self.start_time)
        fps_text = f'FPS: {int(fps)}'
        cv2.putText(im0, fps_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    def start_recording(self, frame):
        self.recording = True
        os.makedirs("recordings", exist_ok=True)
        self.frame_width = frame.shape[1]
        self.frame_height = frame.shape[0]
        self.video_writer = cv2.VideoWriter(f"recordings/recording_{datetime.now().strftime('%Y%m%d%H%M%S')}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (self.frame_width, self.frame_height))

    def stop_recording(self):
        self.recording = False
        self.video_writer.release()
        self.video_writer = None

    def feed_record(self, frame):
        if self.recording:
            self.video_writer.write(frame)

    def __call__(self):
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()

        while True:
            self.start_time = time.time()
            ret, frame = cap.read()
            assert ret

            results = self.predict(frame)
            boxes = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            keep = self.non_max_suppression(boxes, scores, iou_threshold=0.5)

            for i in keep:
                class_id = class_ids[i]
                if class_id in self.classes_to_log:
                    x1, y1, x2, y2 = [int(x) for x in boxes[i]]
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    system_coordinates = f"{center_x},{center_y}"
                    class_name = self.model.names[class_id]

                    if class_name not in self.detected_objects:
                        log_entry = f"{current_time}-\tCoordinates of the Camera Goes here \t-{class_name}"
                        with open("log.txt", "a") as log_file:
                            log_file.write(log_entry + "\n")
                        print(log_entry)
                        self.detected_objects[class_name] = True

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)

                    if class_id == 0:  # Person class
                        self.last_person_detection_time = time.time()
                        if not self.recording:
                            self.start_recording(frame)

            if self.recording and time.time() - self.last_person_detection_time >= self.recording_stop_delay:
                self.stop_recording()

            self.feed_record(frame)
            self.display_fps(frame)
            cv2.imshow('Object Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

detector = ObjectDetection(capture_index=0)
detector()