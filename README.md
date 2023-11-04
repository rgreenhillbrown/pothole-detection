
# Pothole Detection with YOLOv8

Detect potholes from cars/trucks using YOLOv8. This project utilizes the power of YOLOv8, an advanced object detection algorithm, to identify potholes on road surfaces. Our model is trained to recognize various sizes and shapes of potholes, providing an essential tool for road maintenance and safety measures.

## Getting Started

To use this project for detecting potholes, follow the steps below:

1. Clone this repository.
2. Install the required dependencies.
3. Use the provided pre-trained weights or train the model with your dataset.

## Prerequisites

Ensure you have the following installed:

- Python 3.8 or above
- PyTorch 1.8 or above
- Other dependencies listed in `requirements.txt`

## Installation

To set up your environment to run the code, follow these steps:

```
git clone https://github.com/rgreenhillbrown/pothole-detection.git
cd pothole-detection
pip install -r requirements.txt
```

## Usage

To detect potholes in images:

```
python detect.py --weights last.pt --img 640 --conf 0.4 --source data/images
```

For real-time pothole detection:

```
python detect.py --weights last.pt --img 640 --conf 0.4 --source 0  # for webcam
```

For YouTube detection:

```
python detect.py --weights last.pt --img 640 --conf 0.4 --source "https://youtube.com/your-url-here"
```

## Training

To train the model with your dataset:

```
python train.py --img 640 --batch 16 --epochs 50 --data dataset.yaml --weights yolov8.pt
```

Ensure you have your `dataset.yaml` file configured correctly for training.

The dataset used for training is not included here, but using your own data you can train it yourself. 
The .yaml file you use to instruct the model should use the following outline:

```
path: ../
train: train/images
val: valid/images
test: test/images

nc: 1
names: ["pothole"]
```

## Configuration

Modify the `hyp.yaml` to tweak hyperparameters for training.

## Example Detection

Here is an example of the pothole detection in action:

![Pothole Detection Example](detected_frames/pothole_detected_000019.jpg)

## License

This project is licensed under the MIT License - see the `LICENSE.md` file for details.

## Acknowledgments

- The YOLOv8 team for providing an efficient and powerful object detection algorithm.

