# Nagato-Sakura-SMS-Checker

[![GitHub Repo stars](https://img.shields.io/github/stars/AmanoShizukikun/Nagato-Sakura-SMS-Checker?style=social)](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/stargazers)
[![GitHub last commit](https://img.shields.io/github/last-commit/AmanoShizukikun/Nagato-Sakura-SMS-Checker)](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/commits/main)
[![GitHub release](https://img.shields.io/github/v/release/AmanoShizukikun/Nagato-Sakura-SMS-Checker)](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/releases)

\[ 中文 | [English](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/blob/main/assets/docs/README_en.md) | [日本語](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/blob/main/assets/docs/README_jp.md) \]

## 介紹
Nagato-Sakura-SMS-Checker 是「長門櫻計畫」的其中一個分支，是為了解決簡訊詐騙而自製的小型簡訊分類模型，可以分類簡訊類型、判斷簡訊中的電話及網址並且測試網址的響應狀態判斷網站是否安全。

## 公告
- ### 有鑑於用戶反饋 1.0.X 的介面不適合給一般人理解程式偵測的結果，故將開始進行 1.1.0 版本的開發以提供更友善的 GUI 介面。

## 近期變動
### 1.0.5（2024 年 3 月 7 日）
![t2i](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/blob/main/assets/preview/1.0.5.jpg)
### 重要變更
- 【重大】新增黑名單網址數據 data/blacklist.txt。
- 【重大】將 test.py 更名為為 Nagato-Sakura-SMS-Checker-CLI.py 並且更新支援 GUI 版本的所有功能。
- 【重大】將 CUDA 版本和 CPU 版本分離，讓沒有 NVIDIA 顯卡的使用者不用再自己去修改 torch.device 。
### 新增功能
- 【新增】網址黑名單功能，會比對黑名單網址數據如果網址相符會警告使用者。
- 【新增】顯示主機名稱及IP，可以顯示主機名稱、IP 位址、從 IP 位址取得的主機名稱。
- 【新增】顯示憑證日期，可以顯示憑證的起始和結束日期，此功能預設關閉可去設定開啟。
- 【新增】輸入框右鍵選單，可以在簡訊輸入框右鍵貼上了。
- 【更新】版本資訊現在可以顯示當前程式的運行模式。
- 【更新】調整了啟用顯示主機名稱及 IP 以及啟用顯示憑證日期的輸出結果，現在輸出結果更加的整齊。
- 【更新】調整了檢查回應狀態的回覆，現在能清楚的知道是在檢測甚麼。
- 【更新】調整了輸出的順序，現在如果啟用顯示主機名稱以及啟用顯示憑證日期後網址的輸出順序不會亂掉。
- 【修復】修復了 GUI 版本背景的 CMD 輸出結果，現在輸出結果不再出現奇怪的換行。
### 已知問題
- 【錯誤】畫面縮放時，程式的UI圖片的部分不會跟著縮放。

### 1.0.4（2024 年 3 月 1 日）
![t2i](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/blob/main/assets/preview/1.0.4.jpg)
### 重要變更
- 【重大】調整模型配置文件的模型資訊。
- 【重大】添加了更多電信詐騙的相關資料進數據集。
- 【重大】移除黑暗模式改為主題。
### 新增功能
- 【新增】現在能判斷出驗證碼簡訊後將驗證碼輸出。
- 【更新】更新了模型的版本大幅提升了電信詐騙的判斷能力。
- 【更新】版本資訊現在可以顯示GUI版本以及模型版本，並且會隨著語言切換了。
- 【修復】修復了變更語言後輸出的報告不會變更語言的問題。
- 【修復】修復了將有小數點的訊息誤認成網址的錯誤。
### 已知問題
- 【錯誤】畫面縮放時，程式的UI圖片的部分不會跟著縮放。
- 【錯誤】GUI 版本背景的 CMD 輸出結果出現奇怪的換行。

### 1.0.3（2024 年 2 月 27 日）
![t2i](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/blob/main/assets/preview/1.0.3.jpg)
### 重要變更
- 【重大】GUI 大幅度調整了顯示的方式，新增了選單欄。
- 【調整】現在GUI也會顯示完整的報告。
- 【調整】刪除暗黑模式按鈕，改用選單欄開啟。
### 新增功能
- 【新增】開啟 - 開啟已儲存的.json檔查看簡訊結果。
- 【新增】儲存 - 儲存簡訊結果為.json檔。
- 【新增】退出 - 退出GUI。
- 【新增】語言 - 可切換繁體中文、英文、日文。
- 【新增】開啟網站 - 開啟程式的GitHub頁面。
- 【新增】版本資訊 - 顯示程式的當前版本。
- 【修復】短網址類型的網站只能判斷到縮網址的地方，無法判斷縮網址後的網站。
### 已知問題
- 【錯誤】畫面縮放時，程式的UI圖片的部分不會跟著縮放。
- 【錯誤】變更語言後輸出的報告不會變更語言。
- 【錯誤】有機率將有小數點的訊息誤認成網址。

[所有發行版本](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/blob/main/assets/docs/Changelog.md)

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
pip install numpy
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


## GUI 介面
![t2i](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/blob/main/assets/samples/GUI.png)
### 開啟GUI
```shell
python Nagato-Sakura-SMS-Checker-GUI.py
```
### 語言
目前 GUI 支援繁體中文、英文、日文三種語言的快速切換，使用上方 選單欄/語言 即可進行切換。
### 主題
目前 GUI 有七種主題分別為現代淺色、現代深色、緋紅之翼、青色陰影、暗度藏青、可愛次元及次元重生，使用上方 選單欄/主題 進行切換。
![t2i](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/blob/main/assets/samples/themes/Cute_Dimension.png)
### 黑名單
目前 GUI 預設開啟黑名單，當檢測到網址時會自動比對黑名單資料庫，如果不想要顯示或比對黑名單可以在上方 選單欄/設定 選擇開啟或關閉。
![t2i](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/blob/main/assets/samples/example/blacklist.png)

### 主機名稱及IP
目前 GUI 預設關閉主機名稱及IP，如果想要顯示主機名稱及IP可以在上方 選單欄/設定 選擇開啟或關閉。
### 憑證日期
目前 GUI 預設關閉憑證日期，如果想要顯示憑證日期可以在上方 選單欄/設定 選擇開啟或關閉。

## GUI 實際使用範例
### 範例 1
簡訊內容: Myfone提醒您：截止2月29號，您門號尚餘19,985點積分,於今日到期，點擊連結立即兌換獎品！

這種簡訊是台灣現在流行的詐騙簡訊，不但訊息以假亂真他的詐騙網站也做得十分逼真，讓我們一起看看長門櫻的辨識結果:

![t2i](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/blob/main/assets/samples/example/scam_sms.png)

長門櫻成功識別出了詐騙訊息，並且偵測出簡訊中的網址並進行基礎的檢查，這裡我們可以看到網址使用的是http而非https，所以長門櫻發出了警告訊息提示使用者該連結有一定的風險。

### 範例 2
簡訊內容: Hami書城的月讀包「限時下載」一年內會提供超過360本書！會員立即參與投票喜愛的書，有機會抽500元 hamibook.tw/2NQYp

這種簡訊是台灣常見的電信廣告，不管是中華電信、亞太電信都會收到，這個網址還是簡訊有時會有的特殊縮網址，這種高難度的問題長門櫻做得出來嗎?

![t2i](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/blob/main/assets/samples/example/advertise_sms.png)

長門櫻成功識別出了廣告訊息，並且將原本無法透過偵測http及www開頭的網址成功偵測出來，並且轉換成正確的網址並測試網址，看來這個廣告是安全的沒有任何問題。

### 範例 3
簡訊內容: OPEN POINT會員您好，驗證碼為47385，如非本人操作，建議您立即更改密碼。提醒您！勿將密碼、驗證碼交付他人以防詐騙。

這種簡訊是常見的驗證碼簡訊，讓我們看看長門櫻會怎麼做?

![t2i](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/blob/main/assets/samples/example/captcha_sms.png)

長門櫻成功識別出了驗證碼簡訊，但是目前1.0.2版本的長門櫻還無法像 apple 公司那麼厲害可以將簡訊內的驗證碼提取出來，看來長門櫻還需要加油啊。

### 範例 4
簡訊內容: 2023/11/24 14:19 您有0918001824來電1通，提醒您，記得回覆重要電話喔！若已接聽或回電請忽略此則簡訊。

這種簡訊是每個人多少會有的未接來電訊息，長門櫻會怎麼做呢?

![t2i](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/blob/main/assets/samples/example/normal_sms.png)

長門櫻將未接來電訊息分類到了一般訊息，並且正確的讀取到了訊息中的電話號碼，實在是太厲害了讓我們一起為長門櫻鼓掌。

### 範例 EX
簡訊內容: 【亞太電信帳務通知】您本期帳單金額為349元，本期帳單將合併於下期寄送，帳單資訊查詢及線上繳費請利用本公司行動客服APP  、官網會員專區  ；也可利用手機直撥988語音或7-11 ibon繳費，若已繳費無需理會本通知，謝謝。

長門櫻表示剛剛我們做的事實在是太小看她了，要我們試試一次給她兩個或以上的網址，長門櫻啊千萬別逞強啊。

![t2i](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/blob/main/assets/samples/example/two_url_sms.png)

長門櫻成功識別出了亞太電信傳的一般訊息，也成功識別出了兩個網址，並且分別進行安全性測試，亞太電信你網站能弄好嗎? 不是 SSL 問題就是 HOST NOT FOUND 工程師實在是太混了 WWW 。


## 待辦事項
- [ ] **高優先度：**
  - [x] 整合所有提示內容的狀態欄。
  - [x] 統一GUI版本和CMD版本的輸出訊息。
  - [x] 判斷非 http 及 www 開頭的網址。
  - [x] 用户指南。
- [ ] **功能:**
  - [x] GUI版本的深色模式。
  - [x] 電話號碼及網址檢測。
  - [x] 一鍵清除所有內容的按鍵。
  - [x] 黑名單。
  - [ ] 可縮放的調整畫面比例的GUI。


## 致謝
特別感謝以下項目和貢獻者：
### 項目
- [requests](https://github.com/psf/requests)
- [165反詐騙諮詢專線_假投資(博弈)網站](https://data.gov.tw/dataset/160055)
### 貢獻者
<a href="https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=AmanoShizukikun/Nagato-Sakura-SMS-Checker" />
</a>
