# Nagato-Sakura-SMS-Checker

[![GitHub Repo stars](https://img.shields.io/github/stars/AmanoShizukikun/Nagato-Sakura-SMS-Checker?style=social)](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/stargazers)
[![GitHub last commit](https://img.shields.io/github/last-commit/AmanoShizukikun/Nagato-Sakura-SMS-Checker)](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/commits/main)

\[ 中文 | [English](README_en.md) | [日本語](README_jp.md) \]

## 介紹
Nagato-Sakura-SMS-Checker 是「長門櫻計畫」的其中一個分支，是為了解決簡訊詐騙而自製的小型簡訊分類模型，可以分類簡訊類型、判斷簡訊中的電話及網址並且測試網址的響應狀態判斷網站是否安全。

## 公告
- Project-SMS 將在 3.0.0 版本以後正式改名為 「Nagato-Sakura-SMS-Checker」，正式併入「長門櫻計畫」，因該說這個程式早已和「長門櫻」深深的連結在一起了，Project-SMS 將完全移除並從 Nagato-Sakura-SMS-Checker 1.0.0 版本重新開始
- 目前暫時中止 Nagato-Sakura-SMS-Checker 的打包.ex 計畫，直到找到更有效率的的pytorch模型打包方法。

## 近期變動
### 1.0.0（2024 年 2 月 19 日）
![t2i](assets/preview/1.0.0.png)
### 重要變更
- 正式改名為 「Nagato-Sakura-SMS-Checker」
- 微調了訓練的資料檔
- 重新放上了已經訓練好的模型黨

### 新增功能
- 新增判斷簡訊中的電話並列出來
- 新增判斷簡訊中的網址並列出來，並且可以同時測試網址的響應狀態來判斷該網址是否安全
- 新增狀態欄，避免了狀態碼超出GUI的問題
- 新增深色模式，現在可以在淺色模式與深色模式之間切換了

### 已知問題
- 畫面縮放程式的UI不會跟著縮放

## 快速開始
 **粗體** 的是強制要求的。
 
### 硬體要求
1. 作業系統：Windows
1. **CPU** / Nvidia GPU

### 環境設置
- **Python 3**
- 下載: https://www.python.org/downloads/windows/
- **PyTorch**
- 下載: https://pytorch.org/
- NVIDIA GPU驅動程式
- 下載: https://www.nvidia.com/zh-tw/geforce/drivers/
- NVIDIA CUDA Toolkit
- 下載: https://developer.nvidia.com/cuda-toolkit
- NVIDIA cuDNN
- 下載: https://developer.nvidia.com/cudnn

### 檔案說明
- 必要檔案
  main.py:訓練程式
  test.py:測試程式
  SMS_data.json:訓練資料庫
  
- 附加檔案(透過main.py生成)
  config.json:模型配置文件
  labels.txt:標籤文件
  SMS_model.bin:模型
  tokenizer.json:詞彙表

### 安裝
```shell
git clone https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker.git
cd Nagato-Sakura-SMS-Checker
```

- 修改訓練資料庫
```shell
.\SMS_data.json
```

- 開始訓練
```shell
python main.py
```

- 開始測試
```shell
python test.py
```


### GUI
![t2i](assets/samples/light_mode.png)
![t2i](assets/samples/dark_mode.png)
![t2i](assets/samples/test.png)
