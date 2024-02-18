# Nagato-Sakura-SMS-Checker

\[ 中文 | [English](README_en.md) | [日本語](README_jp.md) \]

Nagato-Sakura-SMS-Checker 是「長門櫻計畫」的其中一個分支，是為了解決簡訊詐騙而自製的小型簡訊分類模型，可以分類簡訊類型、判斷簡訊中的電話及網址並且測試網址的響應狀態判斷網站是否安全。

# 公告
- Project-SMS 將在 3.0.0 版本以後正式改名為 「Nagato-Sakura-SMS-Checker」，正式併入「長門櫻計畫」，因該說這個程式早已和「長門櫻」深深的連結在一起了。
- Project-SMS 將完全移除並從 Nagato-Sakura-SMS-Checker 1.0.0 版本重新開始

## 近期變動
### 1.0.0（2024 年 2 月 18 日）
### 重要變更

### 新增功能



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
git clone https://github.com/AmanoShizukikun/Project-SMS.git
cd Project-SMS
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


### 懶人安裝
直接運行run.bat
