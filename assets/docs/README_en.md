# Nagato-Sakura-SMS-Checker

[![GitHub Repo stars](https://img.shields.io/github/stars/AmanoShizukikun/Nagato-Sakura-SMS-Checker?style=social)](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/stargazers)
[![GitHub last commit](https://img.shields.io/github/last-commit/AmanoShizukikun/Nagato-Sakura-SMS-Checker)](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/commits/main)
[![GitHub release](https://img.shields.io/github/v/release/AmanoShizukikun/Nagato-Sakura-SMS-Checker)](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/releases)

\[ [中文](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/blob/main/README.md) | English  | [日本語](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/blob/main/assets/docs/README_jp.md) \]

## Introduction
Nagato-Sakura-SMS-Checker is one of the branches of the "Nagato Sakura Project," created to combat SMS scams. It is a small-scale SMS classification model designed to categorize types of SMS, identify phone numbers and URLs within SMS, and test the response status of URLs to determine if a website is secure.

## Announcements
- ### Project-SMS has officially been renamed to "Nagato-Sakura-SMS-Checker" and merged into the "Nagato Sakura Project." Old files from Project-SMS (prior to version 3.0.0) will be completely removed. Nagato-Sakura-SMS-Checker starts fresh from version 1.0.0.
- ### The plan to package Nagato-Sakura-SMS-Checker into a .exe file is currently suspended until a more efficient PyTorch model packaging method is found.

## Recent Changes
### 1.0.3 (February 27, 2024)
![t2i](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/blob/main/assets/preview/1.0.3.jpg)
### Major Changes
- 【Critical】The GUI has undergone significant adjustments in display, with the addition of a menu bar.
- 【Adjusted】The GUI now displays complete reports.
- 【Adjusted】The dark mode button has been removed and replaced with a menu bar option.
### New Features
- 【Added】Open - Open a saved .json file to view SMS results.
- 【Added】Save - Save SMS results as a .json file.
- 【Added】Exit - Exit the GUI.
- 【Added】Language - Switch between Traditional Chinese, English, and Japanese.
- 【Added】Open Website - Open the program's GitHub page.
- 【Added】Version Info - Display the current version of the program.
- 【Fixed】Websites with shortened URLs could only be detected up to the shortened part, unable to determine the website after expanding the URL.
### Known Issues
- 【Error】When scaling the screen, parts of the program's UI images do not scale accordingly.

### 1.0.2 (February 23, 2024)
![t2i](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/blob/main/assets/preview/1.0.2.jpg)
### Major Changes
- 【Adjusted】The GUI display has been unified to use scrolling text boxes for messages, making it more concise and visually appealing.
- 【Adjusted】The display of GUI and general test program in the terminal has been adjusted, resulting in neater results for users.
- 【Adjusted】Deleted redundant information from the model training data and slightly adjusted training parameters, significantly improving prediction accuracy compared to the previous version.
### New Features
- 【Added】Now capable of detecting websites not starting with "http" or "www" and automatically converting shortened URLs to the original correct URL.
### Known Issues
- 【Error】Websites with shortened URLs could only be detected up to the shortened part, unable to determine the website after expanding the URL.
- 【Error】When scaling the screen, the program's UI does not scale accordingly.

### 1.0.1 (February 21, 2024)
![t2i](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/blob/main/assets/preview/1.0.1.jpg)
### Major Changes
- 【Adjusted】Changed the loading order of models in the GUI and added threads for URL checking, significantly improving GUI responsiveness.
- 【Adjusted】Improved URL detection method, now able to establish SSL/TLS connections, retrieve certificates, and check for suspicious patterns in URL paths.
- 【Adjusted】Dark mode button color adjustments for dark mode.
### New Features
- 【Added】Clear button, allowing users to clear URLs with one click, greatly enhancing usability.
### Known Issues
- 【Error】When scaling the screen, the program's UI does not scale accordingly.

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
  - train.py: Training program
  - test.py: Testing program (CMD Ver.)
  - Nagato-Sakura-SMS-Checker-GUI:Testing program (GUI Ver.)
  - SMS_data.json: Training database
  
- Additional Files (Generated via train.py)
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
python train.py
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

### GUI Theme
Currently, the GUI has two themes: light mode and dark mode. You can switch between them using the button in the bottom right corner.

![t2i](assets/samples/two_mode.png)

## GUI Usage Examples
### Example 1
SMS Content: This week, 6164 Huaxing made a profit of 40%, and next week's strong stock has been selected. Hurry up and add LINE to claim:

This type of SMS is a common investment scam in Taiwan, enticing investors to join LINE groups for fraud. Let's take a look at Nagato Sakura's recognition results:

![t2i](assets/samples/scam_sms.png)

Nagato Sakura successfully identified the suspicious message and detected the URLs in the SMS for basic checks. Here, we can see that the URL uses http instead of https, so Nagato Sakura issued a warning message to alert users to the potential risks of the link.

### Example 2
SMS Content: Hami Bookstore's monthly reading package "Limited Time Download" will provide over 360 books in a year! Members can immediately participate in voting for their favorite books, with a chance to win 500 yuan at hamibook.tw/2NQYp.

This type of SMS is a common telecom advertisement in Taiwan, received by both Chunghwa Telecom and Asia Pacific Telecom. Sometimes, the URL in such messages is a special shortened URL in the SMS. Can Nagato Sakura handle this challenging task?

![t2i](assets/samples/advertise_sms.png)

Nagato Sakura successfully identified the advertising message and managed to detect the URLs that couldn't be detected through the detection of URLs starting with http and www. It then converted them into correct URLs and tested them. It seems that this advertisement is safe without any problems.

### Example 3
SMS Content: OPEN POINT member, your verification code is 47385. If this was not initiated by you, we recommend you change your password immediately. Reminder! Do not disclose your password or verification code to others to prevent fraud.

This type of SMS is a common verification code SMS. Let's see how Nagato Sakura handles it.

![t2i](assets/samples/captcha_sms.png)

Nagato Sakura successfully identified the verification code SMS. However, in version 1.0.2, Nagato Sakura cannot extract the verification code from the SMS like Apple can. It seems that Nagato Sakura still needs to work harder.

### Example 4
SMS Content: 2023/11/24 14:19 You have missed call from 0918001824, reminding you to reply to important calls! If you have answered or returned the call, please ignore this SMS.

This type of SMS is a missed call message that almost everyone receives. How does Nagato Sakura handle it?

![t2i](assets/samples/normal_sms.png)

Nagato Sakura categorized the missed call message as a normal message and correctly read the phone number from the message. It's really impressive. Let's give Nagato Sakura a round of applause.

### Example EX
SMS Content: [Asia Pacific Telecom Billing Notice] Your current bill amount is 349 yuan. This bill will be combined and sent in the next period. For bill inquiries and online payment, please use our company's mobile customer service APP, official website member area; you can also use your mobile phone to directly dial 988 for voice or pay at 7-11 ibon. If the payment has been made, please disregard this notice. Thank you.

Nagato Sakura says that what we just did is underestimating her. She wants us to try giving her two or more URLs. Nagato Sakura, please don't be too hard on yourself.

![t2i](assets/samples/two_sms.png)

Nagato Sakura successfully identified the general message from Asia Pacific Telecom and also identified two URLs. It then performed security tests on them separately. Asia Pacific Telecom, can your website be fixed? It's either an SSL issue or a HOST NOT FOUND issue. Your engineers are really confused with the WWW.


## To-Do List
- [ ] **High Priority:**
  - [x] Integrate status bar for all prompts.
  - [x] Standardize output messages for GUI and CMD versions.
  - [x] Detect URLs not starting with http or www.
  - [x] User guide.

- [ ] **Features:**
  - [x] Dark mode for GUI version.
  - [x] Detection of phone numbers and URLs.
  - [x] Button to clear all content with one click.
  - [ ] Scalable GUI for adjusting screen proportions.
  - [ ] Phone number blacklist.

## Acknowledgments
Special thanks to the following projects and contributors:

### Projects
- [requests](https://github.com/psf/requests)

### Contributors
<a href="https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=AmanoShizukikun/Nagato-Sakura-SMS-Checker" />
</a>
