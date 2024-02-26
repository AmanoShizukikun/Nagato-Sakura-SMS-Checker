# Nagato-Sakura-SMS-Checker

[![GitHub Repo stars](https://img.shields.io/github/stars/AmanoShizukikun/Nagato-Sakura-SMS-Checker?style=social)](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/stargazers)
[![GitHub last commit](https://img.shields.io/github/last-commit/AmanoShizukikun/Nagato-Sakura-SMS-Checker)](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/commits/main)
[![GitHub release](https://img.shields.io/github/v/release/AmanoShizukikun/Nagato-Sakura-SMS-Checker)](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/releases)

\[ [中文](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/blob/main/README.md) | [English](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/blob/main/assets/docs/README_en.md) | 日本語 \]

## 紹介
長門桜SMSチェッカーは、「長門桜計画」の一つで、SMS詐欺対策のために作成されたものです。これは、SMSの種類を分類し、SMS内の電話番号とURLを特定し、URLの応答ステータスをテストしてウェブサイトが安全かどうかを判断する小規模なSMS分類モデルです。

## お知らせ
- ### プロジェクトSMSは正式に「長門桜SMSチェッカー」と改名され、「長門桜計画」に統合されました。プロジェクトSMS（バージョン3.0.0より前）の古いファイルは完全に削除されます。長門桜SMSチェッカーはバージョン1.0.0から新たに始まります。
- ### 長門桜SMSチェッカーを.exeファイルにパッケージ化する計画は、より効率的なPyTorchモデルのパッケージ化方法が見つかるまで一時停止しています。

## 最近の変更
### 1.0.2（2024年2月23日）
![t2i](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/blob/main/assets/preview/1.0.2.jpg)
### 重要な変更
- [調整] GUIの表示方法を調整し、メッセージをスクロールするテキストボックスを一貫して使用するようにしました。これにより、よりすっきりとした見た目になりました。
- [調整] 一般的なテストのために、GUIおよび端末の表示を調整し、結果がより整然となり、ユーザーが読み取りやすくなりました。
- [調整] モデルのトレーニングデータから不要な情報を削除し、トレーニングパラメーターを微調整しました。予測精度は以前のバージョンと比較して明らかに向上しています。
### 新機能
- [新規] 「非」httpおよびwwwから始まるURLを認識し、短縮URLを元の正しい完全な形式に自動変換する機能を追加しました。
###既知の問題
- [エラー] 短縮URLタイプのウェブサイトでは、短縮URLの部分しか認識できず、短縮URLの後のウェブサイトは認識できません。
- [エラー] 画面のズーム時、プログラムのUIがズームに追従しません。

### 1.0.1（2024年2月21日）
![t2i](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/blob/main/assets/preview/1.0.1.jpg)
### 重要な変更
- [調整] GUIはモデルの読み込み順序を調整し、URLのチェックのためにスレッドを追加し、GUIの応答性を大幅に向上させました。
- [調整] URLの検出方法を改善し、SSL/TLS接続の確立、証明書の取得、およびURLパスの可疑なパターンのチェックが可能になりました。
- [調整] ダークモードでは、ボタンの色を調整しました。
### 新機能
- [新規] クリアボタンを追加し、プログラムでURLを一括クリアできるようにし、使いやすさを大幅に向上させました。
### 既知の問題
- [エラー] 画面を拡大縮小すると、プログラムのUIが適切に拡大縮小されません。

### 1.0.0（2024年2月19日）
![t2i](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/blob/main/assets/preview/1.0.0.jpg)
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

### GUI テーマ
現在、GUI にはライトモードとダークモードの2つのテーマがあります。右下隅のボタンを使用して切り替えることができます。

![t2i](assets/samples/two_mode.png)

## GUI 使用例
### 例1
SMS の内容：今週、6164華興は40％の利益を上げ、来週の強力な株式が選ばれました。LINEに追加して請求してください：

この種のSMSは、台湾でよく見られる投資詐欺です。投資家をLINEグループに誘導して詐欺を行います。 長門サクラの認識結果を見てみましょう：

![t2i](assets/samples/scam_sms.png)

長門サクラは、疑わしいメッセージを正常に識別し、SMS内のURLを検出して基本的なチェックを行いました。ここでは、URLがhttpsではなくhttpを使用しているため、長門サクラはユーザーにリンクの潜在的なリスクを警告するメッセージを発行しました。

### 例2
SMS の内容：Hami書店の月間読書パッケージ「期間限定ダウンロード」は、1年で360冊以上の書籍を提供します！メンバーはすぐにお気に入りの本に投票することができ、hamibook.tw/2NQYpで500元を獲得するチャンスがあります。

この種のSMSは、台湾でよく見られる電気通信会社の広告です。中華電信と亜太電信の両方が受け取ります。 このようなメッセージのURLは、時々SMS内に特別な短縮URLであることがあります。この難しいタスクを長門サクラが処理できるでしょうか？

![t2i](assets/samples/advertise_sms.png)

長門サクラは、広告メッセージを正常に識別し、httpやwwwで始まるURLを検出できなかったURLを正常に検出し、それらを正しいURLに変換し、それらをテストしました。この広告は問題なく安全であるようです。

### 例3
SMS の内容：OPEN POINT会員の皆様、確認コードは47385です。これが本人によるものでない場合は、すぐにパスワードを変更することをお勧めします。注意！パスワードや確認コードを他人に漏らさないでください。

この種のSMSは一般的な確認コードのSMSです。長門サクラはこれをどのように処理するのか見てみましょう。

![t2i](assets/samples/captcha_sms.png)

長門サクラは、確認コードのSMSを正常に識別しました。ただし、バージョン1.0.2では、AppleのようにSMSから確認コードを抽出することができません。長門サクラはまだ頑張る必要があるようです。

### 例4
SMS の内容：2023/11/24 14:19 あなたは0918001824からの未着信1通があります。重要な電話に返信することをお忘れなく！ 既に通話を受けたり返信した場合は、このSMSを無視してください。

この種のSMSは、ほとんどの人が受け取る未着信のメッセージです。長門サクラはこれをどのように処理するのでしょうか？

![t2i](assets/samples/normal_sms.png)

長門サクラは、未着信のメッセージを一般的なメッセージとして分類し、メッセージから正しく電話番号を読み取りました。本当に素晴らしいです。長門サクラに拍手を送りましょう。

### 例 EX
SMS の内容：【亜太電信請求通知】現在の請求額は349元です。この請求書は次の期間に統合されて送信されます。請求書のお問い合わせやオンライン支払いには、当社のモバイル顧客サービスAPP、公式ウェブサイトの会員エリアをご利用ください。また、携帯電話で直接988にダイヤルしたり、7-11 ibonで支払いをすることもできます。支払いが完了した場合は、この通知を無視してください。ありがとうございます。

長門サクラは、私たちがやったことは自分を過小評価していると言います。彼女に2つ以上のURLを与えてみるように私たちに頼みます。長門サクラ、無理をしないでください。

![t2i](assets/samples/two_sms.png)

長門サクラは、亜太電信からの一般的なメッセージを正常に識別し、2つのURLも正常に識別し、それぞれにセキュリティテストを実施しました。亜太電信、あなたのウェブサイトを修正できますか？SSLの問題でもホストが見つかりませんの問題でもありません。エンジニアは本当にWWWで混乱しています。


## タスクリスト
- [ ] **高優先度：**
  - [x] すべてのプロンプトのステータスバーを統合する。
  - [x] GUIバージョンとCMDバージョンの出力メッセージを統一する。
  - [x] httpやwwwではじまらないURLを検出する。
  - [x] ユーザーガイド。

- [ ] **機能:**
  - [x] GUIバージョンのダークモード。
  - [x] 電話番号とURLの検出。
  - [x] すべてのコンテンツをクリアするボタン。
  - [ ] リサイズ可能な画面比率のGUI。
  - [ ] 電話番号のブラックリスト。

## 謝辞
以下のプロジェクトと貢献者に特別な感謝を表します：

- [requests](https://github.com/psf/requests)

## 全ての貢献者に感謝いたします

<a href="https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=AmanoShizukikun/Nagato-Sakura-SMS-Checker" />
</a>