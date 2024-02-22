# Nagato-Sakura-SMS-Checker

[![GitHub Repo stars](https://img.shields.io/github/stars/AmanoShizukikun/Nagato-Sakura-SMS-Checker?style=social)](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/stargazers)
[![GitHub last commit](https://img.shields.io/github/last-commit/AmanoShizukikun/Nagato-Sakura-SMS-Checker)](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/commits/main)
[![GitHub release](https://img.shields.io/github/v/release/AmanoShizukikun/Nagato-Sakura-SMS-Checker)](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/releases)

\[ [中文](README.md) | [English](README_en.md) | 日本語 \]

## 紹介
長門桜SMSチェッカーは、「長門桜計画」の一つで、SMS詐欺対策のために作成されたものです。これは、SMSの種類を分類し、SMS内の電話番号とURLを特定し、URLの応答ステータスをテストしてウェブサイトが安全かどうかを判断する小規模なSMS分類モデルです。

## お知らせ
- ### プロジェクトSMSは正式に「長門桜SMSチェッカー」と改名され、「長門桜計画」に統合されました。プロジェクトSMS（バージョン3.0.0より前）の古いファイルは完全に削除されます。長門桜SMSチェッカーはバージョン1.0.0から新たに始まります。
- ### 長門桜SMSチェッカーを.exeファイルにパッケージ化する計画は、より効率的なPyTorchモデルのパッケージ化方法が見つかるまで一時停止しています。

## 最近の変更
### 1.0.2（2024年2月23日）
![t2i](assets/preview/1.0.2.png)
### 重要な変更
- [調整] GUIの表示方法を調整し、メッセージをスクロールするテキストボックスを一貫して使用するようにしました。これにより、よりすっきりとした見た目になりました。
- [調整] 一般的なテストのために、GUIおよび端末の表示を調整し、結果がより整然となり、ユーザーが読み取りやすくなりました。
### 新機能
- [新規] 「非」httpおよびwwwから始まるURLを認識し、短縮URLを元の正しい完全な形式に自動変換する機能を追加しました。
###既知の問題
- [エラー] 短縮URLタイプのウェブサイトでは、短縮URLの部分しか認識できず、短縮URLの後のウェブサイトは認識できません。
- [エラー] 画面のズーム時、プログラムのUIがズームに追従しません。

### 1.0.1（2024年2月21日）
![t2i](assets/preview/1.0.1.png)
### 重要な変更
- [調整] GUIはモデルの読み込み順序を調整し、URLのチェックのためにスレッドを追加し、GUIの応答性を大幅に向上させました。
- [調整] URLの検出方法を改善し、SSL/TLS接続の確立、証明書の取得、およびURLパスの可疑なパターンのチェックが可能になりました。
- [調整] ダークモードでは、ボタンの色を調整しました。
### 新機能
- [新規] クリアボタンを追加し、プログラムでURLを一括クリアできるようにし、使いやすさを大幅に向上させました。
### 既知の問題
- [エラー] 画面を拡大縮小すると、プログラムのUIが適切に拡大縮小されません。

### 1.0.0（2024年2月19日）
![t2i](assets/preview/1.0.0.png)
### 重要な変更
- [重要] プロジェクトは正式に「Nagato-Sakura-SMS-Checker」に改名され、Project-SMS 3.0.0（3.0.0を含む）以前の古いバージョンファイルが削除されました。
- [重要] 事前にトレーニングされたモデルファイルを再アップロードしました。
- [調整] トレーニングデータファイルを調整し、SMSの判断に影響する可能性のある関連ないコンテンツを削除しました。
### 新機能
- [新規] SMSコンテンツ検出機能を追加し、SMSから電話番号（台湾限定）とURLをリストアップできるようにしました。
- [新規] URL検出機能を追加し、ウェブサイトの応答状態をテストしてURLの安全性を判断できるようにしました。
- [新規] ステータスバーを追加し、テストされたウェブサイトのステータスコードがGUIの境界を超える問題を回避しました。
- [新規] ダークモードを追加し、プログラムはライトモードとダークモードの間を切り替えることができます。
### 既知の問題
- [エラー] 画面を拡大縮小すると、プログラムのUIが適切に拡大縮小されません。

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
  - train.py：トレーニングプログラム
  - test.py：テストプログラム (CMD Ver.)
  - Nagato-Sakura-SMS-Checker-GUI:テストプログラム (GUI Ver.)
  - SMS_data.json：トレーニングデータベース
  
- その他のファイル（train.pyによって生成）
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
python train.py
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