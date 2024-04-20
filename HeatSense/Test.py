from ultralytics import YOLO
import supervision as sv
import os
import glob

model = YOLO("HeatSense/Heat.pt")
print(model.names)
results = model(source= 0, show = True, save=True,conf= 0.50)