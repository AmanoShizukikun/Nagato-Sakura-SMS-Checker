# Nagato-Sakura-SMS-Checker

[![GitHub Repo stars](https://img.shields.io/github/stars/AmanoShizukikun/Nagato-Sakura-SMS-Checker?style=social)](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/stargazers)
[![GitHub last commit](https://img.shields.io/github/last-commit/AmanoShizukikun/Nagato-Sakura-SMS-Checker)](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/commits/main)
[![GitHub release](https://img.shields.io/github/v/release/AmanoShizukikun/Nagato-Sakura-SMS-Checker)](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/releases)

\[ [中文](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/blob/main/README.md) | English  | [日本語](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/blob/main/assets/docs/README_jp.md) \]

## Introduction
Nagato-Sakura-SMS-Checker is one of the branches of the "Nagato Sakura Project," created to combat SMS scams. It is a small-scale SMS classification model designed to categorize types of SMS, identify phone numbers and URLs within SMS, and test the response status of URLs to determine if a website is secure.

## Announcements

## Recent Changes
### 1.0.5 (March 7, 2024)
![t2i](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/blob/main/assets/preview/1.0.5.jpg)
### Major Changes
- [Critical] Added blacklist URL data data/blacklist.txt.
- [Critical] Renamed test.py to Nagato-Sakura-SMS-Checker-CLI.py and updated to support all features of the GUI version. (All features are now enabled by default)
- [Critical] Separated CUDA version and CPU version, so users without NVIDIA graphics cards no longer need to modify torch.device themselves.
### New Features
- [New] Added blacklist URL functionality, which warns users if the URL matches the blacklist data.
- [New] Display hostname and IP, showing hostname, IP address, and hostname obtained from the IP address.
- [New] Display certificate dates, showing the start and end dates of the certificate. This feature is disabled by default and can be enabled in settings.
- [New] Input box right-click menu, enabling right-click paste in the SMS input box.
- [Update] Version information now displays the current program's running mode.
- [Update] Adjusted the output results for enabling the display of hostname and IP, and enabling the display of certificate dates, making the output results neater.
- [Update] Adjusted response status check replies, making it clearer what is being detected.
- [Update] Adjusted the output order, now the output order of URLs does not get messed up when enabling display of hostname and enabling display of certificate dates. (Only guaranteed for URLs 1~2)
- [Fix] Fixed GUI version background CMD output results, now the output results no longer have strange line breaks.
### Known Issues
- [Error] When scaling the screen, the UI images of the program do not scale accordingly.

### 1.0.4 (March 1, 2024)
![t2i](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/blob/main/assets/preview/1.0.4.jpg)
### Major Changes
- [Critical] Adjusted model configuration file model information.
- [Critical] Added more data related to telecommunications fraud to the dataset.
- [Critical] Removed dark mode and switched to themes.
### New Features
- [New] Now able to detect verification code SMS and output the verification code.
- [Update] Updated the model version significantly improving the ability to detect telecommunications fraud.
- [Update] Version information can now display GUI version and model version, and will change with language switching.
- [Fix] Fixed the issue where the language of the output report did not change after changing the language.
- [Fix] Fixed the issue of mistakenly recognizing messages with decimals as URLs.
### Known Issues
- [Error] When scaling the screen, the UI images of the program do not scale accordingly.
- [Error] Strange line breaks appear in CMD output results of the GUI version background.

### 1.0.3 (February 27, 2024)
![t2i](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/blob/main/assets/preview/1.0.3.jpg)
### Major Changes
- [Critical] GUI significantly adjusted display, adding a menu bar.
- [Adjustment] GUI now displays complete reports.
- [Adjustment] Removed dark mode button, replaced with a menu bar.
### New Features
- [New] Open - Open saved .json file to view SMS results.
- [New] Save - Save SMS results as .json file.
- [New] Exit - Exit GUI.
- [New] Language - Switch between Traditional Chinese, English, and Japanese.
- [New] Open Website - Open program's GitHub page.
- [New] Version Information - Display program's current version.
- [Fix] Fixed issue where URL shortener type websites could only be identified up to the shortened URL and could not identify the website after shortening.
### Known Issues
- [Error] When scaling the screen, the UI images of the program do not scale accordingly.
- [Error] Output report does not change language after changing language.
- [Error] There is a chance of mistakenly recognizing messages with decimals as URLs.

[All Releases](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/blob/main/assets/docs/Changelog.md)

## Quick Start
**Items in bold are mandatory requirements.**
### System Requirements
- System Requirements: 64-bit Windows
- **Processor**: 64-bit processor
- **Memory**: 2GB
- Graphics Card: NVIDIA graphics card with 1GB VRAM and CUDA acceleration support
- **Storage**: 3GB available space

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
- Python
```shell
pip install Pillow
pip install requests
pip install numpy
```

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


## GUI
![t2i](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/blob/main/assets/samples/GUI.png)
- Launch the GUI
```shell
python Nagato-Sakura-SMS-Checker-GUI.py
```

### Language
Currently, the GUI supports quick switching between Traditional Chinese, English, and Japanese. Simply use the top menu bar/Language to switch.

### Themes
Currently, the GUI offers seven themes: Modern Light, Modern Dark, Crimson Wing, Blue Shade, Dark Indigo, Cute Dimension, and Dimension Rebirth. Switch between them using the top menu bar/Themes.
![t2i](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/blob/main/assets/samples/themes/Cute_Dimension.png)


## GUI Practical Usage Example
### Example 1
SMS Content: Myfone Reminder: As of February 29, you have 19,985 points remaining in your number. Expires today. Click the link to redeem your prize!

This type of SMS is a popular scam SMS in Taiwan. Not only is the message deceptive, but the scam website is also very convincing. Let's take a look at Nagato Sakura's identification result:

![t2i](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/blob/main/assets/samples/example/scam_sms.png)

Nagato Sakura successfully identifies the scam message and detects the URL in the SMS for basic inspection. Here, we can see that the URL uses HTTP instead of HTTPS, so Nagato Sakura issues a warning message to alert the user of potential risks.

### Example 2
SMS Content: Hami Bookstore's monthly reading package "Limited Time Download" will provide over 360 books in a year! Members can immediately participate in voting for their favorite books, with a chance to win 500 yuan at hamibook.tw/2NQYp.

This type of SMS is a common telecom advertisement in Taiwan, received by both Chunghwa Telecom and Asia Pacific Telecom. Sometimes, the URL in such messages is a special shortened URL in the SMS. Can Nagato Sakura handle this challenging task?

![t2i](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/blob/main/assets/samples/example/advertise_sms.png)

Nagato Sakura successfully identified the advertising message and managed to detect the URLs that couldn't be detected through the detection of URLs starting with http and www. It then converted them into correct URLs and tested them. It seems that this advertisement is safe without any problems.

### Example 3
SMS Content: OPEN POINT member, your verification code is 47385. If this was not initiated by you, we recommend you change your password immediately. Reminder! Do not disclose your password or verification code to others to prevent fraud.

This type of SMS is a common verification code SMS. Let's see how Nagato Sakura handles it.

![t2i](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/blob/main/assets/samples/example/captcha_sms.png)

Nagato Sakura successfully identified the verification code SMS. However, in version 1.0.2, Nagato Sakura cannot extract the verification code from the SMS like Apple can. It seems that Nagato Sakura still needs to work harder.

### Example 4
SMS Content: 2023/11/24 14:19 You have missed call from 0918001824, reminding you to reply to important calls! If you have answered or returned the call, please ignore this SMS.

This type of SMS is a missed call message that almost everyone receives. How does Nagato Sakura handle it?

![t2i](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/blob/main/assets/samples/example/normal_sms.png)

Nagato Sakura categorized the missed call message as a normal message and correctly read the phone number from the message. It's really impressive. Let's give Nagato Sakura a round of applause.

### Example EX
SMS Content: [Asia Pacific Telecom Billing Notice] Your current bill amount is 349 yuan. This bill will be combined and sent in the next period. For bill inquiries and online payment, please use our company's mobile customer service APP, official website member area; you can also use your mobile phone to directly dial 988 for voice or pay at 7-11 ibon. If the payment has been made, please disregard this notice. Thank you.

Nagato Sakura says that what we just did is underestimating her. She wants us to try giving her two or more URLs. Nagato Sakura, please don't be too hard on yourself.

![t2i](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/blob/main/assets/samples/example/two_url_sms.png)

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
- [165反詐騙諮詢專線_假投資(博弈)網站](https://data.gov.tw/dataset/160055)

### Contributors
<a href="https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=AmanoShizukikun/Nagato-Sakura-SMS-Checker" />
</a>
