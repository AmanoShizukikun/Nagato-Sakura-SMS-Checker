# Nagato-Sakura-SMS-Checker

[![GitHub Repo stars](https://img.shields.io/github/stars/AmanoShizukikun/Nagato-Sakura-SMS-Checker?style=social)](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/stargazers)
[![GitHub last commit](https://img.shields.io/github/last-commit/AmanoShizukikun/Nagato-Sakura-SMS-Checker)](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/commits/main)
[![GitHub release](https://img.shields.io/github/v/release/AmanoShizukikun/Nagato-Sakura-SMS-Checker)](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/releases)

\[ 中文 | [English](README_en.md) | [日本語](README_jp.md) \]

## 介紹
Nagato-Sakura-SMS-Checker 是「長門櫻計畫」的其中一個分支，是為了解決簡訊詐騙而自製的小型簡訊分類模型，可以分類簡訊類型、判斷簡訊中的電話及網址並且測試網址的響應狀態判斷網站是否安全。

## 公告
- ### Project-SMS 正式改名為 「Nagato-Sakura-SMS-Checker」，並且併入「長門櫻計畫」，Project-SMS (3.0.0以前)的舊檔案將完全移除， Nagato-Sakura-SMS-Checker 版本從 1.0.0 重新開始
- ### 目前中止 Nagato-Sakura-SMS-Checker 的打包 .exe 計畫，直到找到更有效率的的 Pytorch 模型打包方法。

## 近期變動
### 1.0.2
### 1.0.1（2024 年 2 月 21 日）
![t2i](assets/preview/1.0.1.png)
### 重要變更
- 【調整】GUI 調整了模型加載的順序，並且新增了線程來進行網址檢查，大幅提高了 GUI 的反應速度。
- 【調整】改善了網址判斷的方式，現在可以建立 SSL/TLS 連線並取得證書以及檢查網址路徑是否包含可疑模式。
- 【調整】深色模式，調整了深色模式的按鈕的顏色。
### 新增功能
- 【新增】清除鍵，現在程式可以一鍵清除網址，大幅提升了使用的方便性。
### 已知問題
- 【錯誤】畫面縮放時，程式的UI不會跟著縮放。
  
### 1.0.0（2024 年 2 月 19 日）
![t2i](assets/preview/1.0.0.png)
### 重要變更
- 【重大】專案正式改名為 「Nagato-Sakura-SMS-Checker」，並且移除了Project-SMS 3.0.0 (包含3.0.0) 以前的舊版本檔案。
- 【重大】重新上傳了已經訓練好的模型檔。
- 【調整】調整了訓練的資料檔，刪除部分有可能影響到簡訊判斷的無要內容。
### 新增功能
- 【新增】簡訊內容偵測功能，可以將簡訊中的電話(限台灣地區)及網址並列出來。
- 【新增】網址檢測功能，可以測試網站的響應狀態來判斷該網址是否安全。
- 【新增】狀態欄，避免了測試網站的狀態碼超出GUI的問題。
- 【新增】深色模式，現在程式可以在淺色模式與深色模式之間切換了。
### 已知問題
- 【錯誤】畫面縮放時，程式的UI不會跟著縮放。

## 快速開始
 **粗體** 的是強制要求的。
### 系統需求
- 系統需求: 64-bit Windows
- **處理器**: 64 位元的處理器
- **記憶體**: 2GB
- 顯示卡: 1GB VRAM 且支援 CUDA 加速的 NVIDIA 顯示卡
- **儲存空間**: 3GB 可用空間

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
- Python庫
```shell
pip install Pillow
pip install requests
```

### 檔案說明
- 必要檔案
  - train.py:訓練程式
  - test.py:測試程式 (CMD版本)
  - Nagato-Sakura-SMS-Checker-GUI:測試程式 (GUI版本)
  - SMS_data.json:訓練資料庫
  
- 附加檔案(透過train.py生成)
  - config.json:模型配置文件
  - labels.txt:標籤文件
  - SMS_model.bin:模型
  - tokenizer.json:詞彙表

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
python train.py
```

- 開始測試
```shell
python test.py
```


### GUI 介面
- 開啟GUI
```shell
python Nagato-Sakura-SMS-Checker-GUI.py
```

- 右下角的按鈕可以切換深色模式/淺色模式
![t2i](assets/samples/two_mode.png)

## 實際範例
- ### 有網址的一般簡訊
![t2i](assets/samples/test_01.png)

- ### 有電話的一般簡訊
![t2i](assets/samples/test_02.png)

- ### 疑似網址有問題的簡訊
![t2i](assets/samples/test_03.png)
