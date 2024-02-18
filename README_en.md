# Nagato-Sakura-SMS-Checker

[![GitHub Repo stars](https://img.shields.io/github/stars/AmanoShizukikun/Nagato-Sakura-SMS-Checker?style=social)](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/stargazers)
[![GitHub last commit](https://img.shields.io/github/last-commit/AmanoShizukikun/Nagato-Sakura-SMS-Checker)](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/commits/main)

\[ [中文](README.md) | English  | [日本語](README_jp.md) \]

## Introduction
Nagato-Sakura-SMS-Checker is one of the branches of the "Nagato Sakura Project," created to combat SMS scams. It is a small-scale SMS classification model designed to categorize types of SMS, identify phone numbers and URLs within SMS, and test the response status of URLs to determine if a website is secure.

## Announcements
- ### Project-SMS has officially been renamed to "Nagato-Sakura-SMS-Checker" and merged into the "Nagato Sakura Project." Old files from Project-SMS (prior to version 3.0.0) will be completely removed. Nagato-Sakura-SMS-Checker starts fresh from version 1.0.0.
- ### The plan to package Nagato-Sakura-SMS-Checker into a .exe file is currently suspended until a more efficient PyTorch model packaging method is found.

## Recent Changes
### 1.0.0 (February 19, 2024)
![t2i](assets/preview/1.0.0.png)
### Major Changes
- Officially renamed to "Nagato-Sakura-SMS-Checker."
- Fine-tuned training data files.
- Reuploaded pre-trained model files.

### New Features
- Added functionality to identify phone numbers within SMS and display them.
- Added functionality to identify URLs within SMS, display them, and simultaneously test the response status of URLs to determine their safety.
- Added a status bar to prevent overflow of status codes in the GUI.
- Added dark mode, allowing users to switch between light and dark modes.

### Known Issues
- UI does not scale with screen zoom.

## Quick Start
**Items in bold are mandatory requirements.**

### Hardware Requirements
1. Operating System: Windows
2. **CPU** / Nvidia GPU

### Environment Setup
- **Python 3**
- Download: [Python](https://www.python.org/downloads/windows/)
- **PyTorch**
- Download: [PyTorch](https://pytorch.org/)
- NVIDIA GPU Driver
- Download: [NVIDIA Drivers](https://www.nvidia.com/zh-tw/geforce/drivers/)
- NVIDIA CUDA Toolkit
- Download: [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- NVIDIA cuDNN
- Download: [cuDNN](https://developer.nvidia.com/cudnn)

### File Descriptions
- Mandatory Files
  - main.py: Training program
  - test.py: Testing program
  - SMS_data.json: Training database
  
- Additional Files (Generated via main.py)
  - config.json: Model configuration file
  - labels.txt: Label file
  - SMS_model.bin: Model
  - tokenizer.json: Vocabulary

### Installation
```shell
git clone https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker.git
cd Nagato-Sakura-SMS-Checker
```

- Modify the training database
```shell
.\SMS_data.json
```

- Start training
```shell
python main.py
```

- Start testing
```shell
python test.py
```

### GUI
- Launch the GUI
```shell
python Nagato-Sakura-SMS-Checker-GUI.py
```

- The button in the bottom right corner allows toggling between dark mode and light mode.
![t2i](assets/samples/two_mode.png)

## Examples
- ### Normal SMS with URL
![t2i](assets/samples/test_01.png)

- ### Normal SMS with Phone Number
![t2i](assets/samples/test_02.png)

- ### Suspicious SMS with Problematic URL
![t2i](assets/samples/test_03.png)