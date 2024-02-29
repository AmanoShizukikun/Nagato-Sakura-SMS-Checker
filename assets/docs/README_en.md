# Nagato-Sakura-SMS-Checker

[![GitHub Repo stars](https://img.shields.io/github/stars/AmanoShizukikun/Nagato-Sakura-SMS-Checker?style=social)](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/stargazers)
[![GitHub last commit](https://img.shields.io/github/last-commit/AmanoShizukikun/Nagato-Sakura-SMS-Checker)](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/commits/main)
[![GitHub release](https://img.shields.io/github/v/release/AmanoShizukikun/Nagato-Sakura-SMS-Checker)](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/releases)

\[ [中文](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/blob/main/README.md) | English  | [日本語](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/blob/main/assets/docs/README_jp.md) \]

## Introduction
Nagato-Sakura-SMS-Checker is one of the branches of the "Nagato Sakura Project," created to combat SMS scams. It is a small-scale SMS classification model designed to categorize types of SMS, identify phone numbers and URLs within SMS, and test the response status of URLs to determine if a website is secure.

## Announcements
- ### The plan to package Nagato-Sakura-SMS-Checker into a .exe file is currently suspended until a more efficient PyTorch model packaging method is found.

## Recent Changes
### 1.0.4 (March 1, 2024)
![t2i](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/blob/main/assets/preview/1.0.4.jpg)
### Important Changes
- [Critical] Adjusted model configuration file's model information.
- [Critical] Added more data related to telecom fraud to the dataset (current total: 124).
- [Critical] Removed dark mode and replaced it with a theme.
### New Features
- [New] Now capable of identifying verification code SMS and outputting the code.
- [Update] Upgraded the model version significantly improving telecom fraud detection capability.
- [Update] Version information now displays GUI version and model version, and updates with language changes.
- [Fix] Corrected issue where reports weren't changing language after language switch.
- [Fix] Rectified error where messages with decimals were mistakenly identified as URLs.
### Known Issues
- [Error] UI images of the program do not scale with screen zoom.

### 1.0.3 (February 27, 2024)
![t2i](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/blob/main/assets/preview/1.0.3.jpg)
### Important Changes
- [Critical] GUI significantly redesigned with added menu bar.
- [Adjustment] GUI now displays complete reports.
- [Adjustment] Removed dark mode button, replaced with menu bar toggle.
### New Features
- [New] Open - Open saved .json file to view SMS results.
- [New] Save - Save SMS results as .json file.
- [New] Quit - Exit GUI.
- [New] Language - Toggle between Traditional Chinese, English, and Japanese.
- [New] Open Website - Open program's GitHub page.
- [New] Version Info - Display current program version.
- [Fix] Shortened URL websites now correctly identified.
### Known Issues
- [Error] UI images of the program do not scale with screen zoom.
- [Error] Reports do not change language after language switch.
- [Error] Chance of messages with decimals being mistaken for URLs.

### 1.0.2 (February 23, 2024)
![t2i](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/blob/main/assets/preview/1.0.2.jpg)
### Important Changes
- [Adjustment] GUI display redesigned with unified use of scrolling text box for messages, providing a cleaner and more aesthetic appearance.
- [Adjustment] GUI and terminal display for general testing now show results more neatly, facilitating easier user readability.
- [Adjustment] Training data for the model cleared of irrelevant portions and training parameters slightly adjusted, resulting in a significant improvement in prediction accuracy compared to the old version.
### New Features
- [New] Now capable of identifying "non" http and www starting websites, and automatically converting abbreviated URLs to the correct full URLs.
### Known Issues
- [Error] Shortened URL websites only identified up to the shortened part, unable to determine the website after the shortened part.
- [Error] UI does not scale with screen zoom.

[All changes](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/blob/main/assets/docs/Changelog.md)

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

### Language
Currently, the GUI supports quick switching between Traditional Chinese, English, and Japanese. Simply use the top menu bar/Language to switch.

### Themes
Currently, the GUI offers seven themes: Modern Light, Modern Dark, Crimson Wing, Blue Shade, Dark Indigo, Cute Dimension, and Dimension Rebirth. Switch between them using the top menu bar/Themes.
![t2i](assets/samples/Light_Mode.png)

## GUI Practical Usage Example
### Example 1
SMS Content: Myfone Reminder: As of February 29, you have 19,985 points remaining in your number. Expires today. Click the link to redeem your prize!

This type of SMS is a popular scam SMS in Taiwan. Not only is the message deceptive, but the scam website is also very convincing. Let's take a look at Nagato Sakura's identification result:

![t2i](assets/samples/scam_sms.png)

Nagato Sakura successfully identifies the scam message and detects the URL in the SMS for basic inspection. Here, we can see that the URL uses HTTP instead of HTTPS, so Nagato Sakura issues a warning message to alert the user of potential risks.

### Example 2
SMS Content: Hami Bookstore's monthly reading package "Limited Time Download" will provide over 360 books in a year! Members can immediately participate in voting for their favorite books, with a chance to win 500 yuan at hamibook.tw/2NQYp.

This type of SMS is a common telecom advertisement in Taiwan, received by both Chunghwa Telecom and Asia Pacific Telecom. Sometimes, the URL in such messages is a special shortened URL in the SMS. Can Nagato Sakura handle this challenging task?

![t2i](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/blob/main/assets/samples/advertise_sms.png)

Nagato Sakura successfully identified the advertising message and managed to detect the URLs that couldn't be detected through the detection of URLs starting with http and www. It then converted them into correct URLs and tested them. It seems that this advertisement is safe without any problems.

### Example 3
SMS Content: OPEN POINT member, your verification code is 47385. If this was not initiated by you, we recommend you change your password immediately. Reminder! Do not disclose your password or verification code to others to prevent fraud.

This type of SMS is a common verification code SMS. Let's see how Nagato Sakura handles it.

![t2i](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/blob/main/assets/samples/captcha_sms.png)

Nagato Sakura successfully identified the verification code SMS. However, in version 1.0.2, Nagato Sakura cannot extract the verification code from the SMS like Apple can. It seems that Nagato Sakura still needs to work harder.

### Example 4
SMS Content: 2023/11/24 14:19 You have missed call from 0918001824, reminding you to reply to important calls! If you have answered or returned the call, please ignore this SMS.

This type of SMS is a missed call message that almost everyone receives. How does Nagato Sakura handle it?

![t2i](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/blob/main/assets/samples/normal_sms.png)

Nagato Sakura categorized the missed call message as a normal message and correctly read the phone number from the message. It's really impressive. Let's give Nagato Sakura a round of applause.

### Example EX
SMS Content: [Asia Pacific Telecom Billing Notice] Your current bill amount is 349 yuan. This bill will be combined and sent in the next period. For bill inquiries and online payment, please use our company's mobile customer service APP, official website member area; you can also use your mobile phone to directly dial 988 for voice or pay at 7-11 ibon. If the payment has been made, please disregard this notice. Thank you.

Nagato Sakura says that what we just did is underestimating her. She wants us to try giving her two or more URLs. Nagato Sakura, please don't be too hard on yourself.

![t2i](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/blob/main/assets/samples/two_sms.png)

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
