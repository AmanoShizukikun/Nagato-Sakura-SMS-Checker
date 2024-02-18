# Nagato-Sakura-SMS-Checker

[![GitHub Repo stars](https://img.shields.io/github/stars/AmanoShizukikun/Nagato-Sakura-SMS-Checker?style=social)](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/stargazers)
[![GitHub last commit](https://img.shields.io/github/last-commit/AmanoShizukikun/Nagato-Sakura-SMS-Checker)](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/commits/main)

\[ [中文](README.md) | [English](README_en.md) | 日本語 \]

## 紹介
長門桜SMSチェッカーは、「長門桜計画」の一つで、SMS詐欺対策のために作成されたものです。これは、SMSの種類を分類し、SMS内の電話番号とURLを特定し、URLの応答ステータスをテストしてウェブサイトが安全かどうかを判断する小規模なSMS分類モデルです。

## お知らせ
- ### プロジェクトSMSは正式に「長門桜SMSチェッカー」と改名され、「長門桜計画」に統合されました。プロジェクトSMS（バージョン3.0.0より前）の古いファイルは完全に削除されます。長門桜SMSチェッカーはバージョン1.0.0から新たに始まります。
- ### 長門桜SMSチェッカーを.exeファイルにパッケージ化する計画は、より効率的なPyTorchモデルのパッケージ化方法が見つかるまで一時停止しています。

## 最近の変更
### 1.0.0（2024年2月19日）
![t2i](assets/preview/1.0.0.png)
### 主な変更点
- 正式に「長門桜SMSチェッカー」に改名されました。
- トレーニングデータファイルを微調整しました。
- 事前にトレーニングされたモデルファイルを再アップロードしました。

### 新機能
- SMS内の電話番号を特定し、表示する機能を追加しました。
- SMS内のURLを特定し、表示し、同時にURLの応答ステータスをテストして安全性を判断する機能を追加しました。
- GUIのステータスコードのオーバーフローを防ぐためにステータスバーを追加しました。
- ダークモードを追加し、ユーザーがライトモードとダークモードを切り替えることができるようにしました。

###既知の問題
- UIは画面のズームに追随しません。

## クイックスタート
**太字の項目は必須の要件です。**

### ハードウェア要件
1. オペレーティングシステム：Windows
2. **CPU** / Nvidia GPU

### 環境の設定
- **Python 3**
- ダウンロード：[Python](https://www.python.org/downloads/windows/)
- **PyTorch**
- ダウンロード：[PyTorch](https://pytorch.org/)
- NVIDIA GPUドライバー
- ダウンロード：[NVIDIAドライバー](https://www.nvidia.com/zh-tw/geforce/drivers/)
- NVIDIA CUDA Toolkit
- ダウンロード：[CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- NVIDIA cuDNN
- ダウンロード：[cuDNN](https://developer.nvidia.com/cudnn)

### ファイルの説明
- 必須ファイル
  - main.py：トレーニングプログラム
  - test.py：テストプログラム
  - SMS_data.json：トレーニングデータベース
  
- その他のファイル（main.pyによって生成）
  - config.json：モデル構成ファイル
  - labels.txt：ラベルファイル
  - SMS_model.bin：モデル
  - tokenizer.json：ボキャブラリ

### インストール
```shell
git clone https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker.git
cd Nagato-Sakura-SMS-Checker
```

- トレーニングデータベースの修正
```shell
.\SMS_data.json
```

- トレーニングの開始
```shell
python main.py
```

- テストの開始
```shell
python test.py
```

### GUI
- GUIを起動
```shell
python Nagato-Sakura-SMS-Checker-GUI.py
```

- 右下のボタンを使用してダークモードとライトモードを切り替えることができます。
![t2i](assets/samples/two_mode.png)

## 例
- ### URLを含む通常のSMS
![t2i](assets/samples/test_01.png)

- ### 電話番号を含む通常のSMS
![t2i](assets/samples/test_02.png)

- ### 問題のあるURLを含む疑わしいSMS
![t2i](assets/samples/test_03.png)