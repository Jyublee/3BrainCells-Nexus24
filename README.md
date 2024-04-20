# Forest Protection System (FPS) 

## Overview
The Forest Protection System (FPS) is a comprehensive integrated surveillance system designed to safeguard forests against illegal activities such as poaching, waste disposal, and wildfires. Leveraging advanced technologies, including YOLOv8 as the backbone architecture and PyTorch framework, FPS provides real-time monitoring and detection capabilities to protect wildlife habitats and natural ecosystems.

## Features
- **Intruder Detection**: Utilizes a network of cameras powered by Convolutional Neural Networks (CNNs) to detect and deter intruders in restricted forest areas.
- **Boundary Surveillance**: Strategically positioned boundary cameras detect wildlife movement indicating potential escapes into human-populated areas.
- **Animal Deterrent**: Develop an automated system to deploy ulatrasonic sound to funcation as animal deterrents.
- **Thermal Imaging Integration**: Incorporate thermal imaging cameras for nighttime surveillance and in dense foliage where visibility is low.
- **Smart Sound System**: Detects illegal activities such as poaching using audio cues, enhancing forest security measures.
- **Automated Response System**: Develop an automated system to deploy alerts to nearby patrols upon intruder detection, reducing response times.
- **Custom Time Restriction**: A Custom Set Time Frame is incorporated in order to facilitate some needs of public forests. 

## Future Enhancements
- **Facial Recognition**: Incorporates facial recognition capabilities to identify intruders and maintain a database for repeat offenders.
- **Behavior Training**: Incorporates Deep learning to connect behaviours of animals in a perticular region and send alrets when there is a drastic deviation in their recorded behavior.
- **Wildfire Detection**: Monitors forest conditions and identifies potential fire outbreaks to mitigate environmental threats.

## Dependencies
- YOLOv8
- PyTorch
- Other necessary dependencies as per the project requirements

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Jyublee/3BrainCells-Nexus24

2. Run The Program:
   ```bash
   python Main.py

## Configuration

- `capture_index`: Index of the camera to be used for live video feed (default: 0).
- `log_delay`: Time interval (in seconds) between logging events (default: 5.0).
- `recording_stop_delay`: Time interval (in seconds) to stop recording after the last object detection (default: 5.0).
- `iou_threshold`: Intersection over Union (IOU) threshold for non-maximum suppression (NMS) (default: 0.5).
- `classes`: List of classes to detect (default: [14, 15, 16, 17, 18, 19, 21, 22, 23, 0]).

## Usage

1. Run the program:

2. The application will open a GUI window displaying live video feed from the camera(s) connected to the system.

3. Detected objects will be highlighted with bounding boxes and labels in the video feed. Detected events will be logged to a file named `logs.txt`.

4. If an object is detected, the application will start recording the video feed. The recording will stop automatically if no objects are detected for a specified duration.

5. The application will terminate automatically outside the specified time frame (e.g., between 11:00 AM to 2:00 PM).

## License

This project is licensed under the MIT License
