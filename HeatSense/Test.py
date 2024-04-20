from ultralytics import YOLO
import supervision as sv
import os
import glob

model = YOLO("Heat_New.pt")
print(model.names)
results = model(source= "Test_vid.mp4" , show = True, save=True,conf= 0.50)