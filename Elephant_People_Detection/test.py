from ultralytics import YOLO
import supervision as sv
import os
import glob

model = YOLO(r"c:\Users\91964\OneDrive\Desktop\Mini Project\FPS\3BrainCells-Nexus24\Elephant_People_Detection\Elephant.pt")

results = model(source= 0, show = True, save=True,conf= 0.30)
