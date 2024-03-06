# Nagato-Sakura-SMS-Checker

[![GitHub Repo stars](https://img.shields.io/github/stars/AmanoShizukikun/Nagato-Sakura-SMS-Checker?style=social)](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/stargazers)
[![GitHub last commit](https://img.shields.io/github/last-commit/AmanoShizukikun/Nagato-Sakura-SMS-Checker)](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/commits/main)
[![GitHub release](https://img.shields.io/github/v/release/AmanoShizukikun/Nagato-Sakura-SMS-Checker)](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/releases)

\[ [中文](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/blob/main/README.md) | [English](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/blob/main/assets/docs/README_en.md) | 日本語 \]

## 紹介
長門桜SMSチェッカーは、「長門桜計画」の一つで、SMS詐欺対策のために作成されたものです。これは、SMSの種類を分類し、SMS内の電話番号とURLを特定し、URLの応答ステータスをテストしてウェブサイトが安全かどうかを判断する小規模なSMS分類モデルです。

## お知らせ

## 最近の変更
### 1.0.5（2024年3月7日）
![t2i](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/blob/main/assets/preview/1.0.5.jpg)
### 重要な変更
- [重要] ブラックリストURLデータ data/blacklist.txt を追加しました。
- [重要] test.py を Nagato-Sakura-SMS-Checker-CLI.py に名前変更し、GUIバージョンのすべての機能をサポートするように更新しました。
- [重要] CUDAバージョンとCPUバージョンを分離し、NVIDIAグラフィックスカードを持たないユーザーがtorch.deviceを自分で変更する必要がなくなりました。
### 新機能
- [新規] ブラックリストURL機能を追加しました。URLがブラックリストデータと一致する場合、ユーザーに警告します。
- [新規] ホスト名とIPを表示します。ホスト名、IPアドレス、IPアドレスから取得したホスト名を表示します。
- [新規] 証明書の日付を表示します。証明書の開始日と終了日を表示します。この機能はデフォルトで無効ですが、設定で有効にできます。
- [新規] 入力ボックスの右クリックメニューを追加しました。SMS入力ボックスで右クリックして貼り付けることができます。
- [更新] バージョン情報に現在のプログラムの実行モードが表示されるようになりました。
- [更新] ホスト名とIPの表示を有効にし、証明書の日付の表示を有効にすると、出力結果がより整った形に調整されます。
- [更新] 応答ステータスのチェック応答を調整しました。何を検出しているかがより明確になります。
- [更新] 出力の順序を調整しました。ホスト名と証明書の日付の表示を有効にした場合、URLの出力順序が乱れなくなりました。
- [修正] GUIバージョンの背景のCMD出力結果が修正され、出力結果に奇妙な改行が表示されなくなりました。
###既知の問題
- [エラー] 画面を拡大すると、プログラムのUI画像が適切に拡大されません。

### 1.0.4（2024年3月1日）
![t2i](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/blob/main/assets/preview/1.0.4.jpg)
### 重要な変更
- [重要] モデル構成ファイルのモデル情報を調整しました。
- [重要] 電気通信詐欺に関連するデータをデータセットに追加しました。
- [重要] ダークモードを削除し、テーマに切り替えました。
### 新機能
- [新規] 確認コードSMSを検出し、確認コードを出力できるようになりました。
- [更新] モデルのバージョンを更新し、電気通信詐欺の検出能力を大幅に向上させました。
- [更新] バージョン情報にGUIバージョンとモデルバージョンを表示し、言語の切り替えに応じて変更されるようになりました。
- [修正] 言語を変更した後、出力レポートの言語が変更されない問題を修正しました。
- [修正] 小数点を含むメッセージを誤ってURLとして認識する問題を修正しました。
### 既知の問題
- [エラー] 画面を拡大すると、プログラムのUI画像が適切に拡大されません。
- [エラー] GUIバージョンの背景のCMD出力結果に奇妙な改行が表示されます。

### 1.0.3（2024年2月27日）
![t2i](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/blob/main/assets/preview/1.0.3.jpg)
### 重要な変更
- [重要] GUIが大幅に調整され、メニューバーが追加されました。
- [調整] GUIは現在完全なレポートを表示します。
- [調整] ダークモードボタンを削除し、メニューバーに置き換えました。
### 新機能
- [新規] 開く - 保存された.jsonファイルを開いてSMSの結果を表示します。
- [新規] 保存 - SMSの結果を.jsonファイルとして保存します。
- [新規] 終了 - GUIを終了します。
- [新規] 言語 - 繁体字中国語、英語、日本語の間を切り替えます。
- [新規] ウェブサイトを開く - プログラムのGitHubページを開きます。
- [新規] バージョン情報 - プログラムの現在のバージョンを表示します。
- [修正] URL短縮タイプのウェブサイトは、短縮URLの場所までしか識別できず、短縮後のウェブサイトを識別できない問題を修正しました。
### 既知の問題
- [エラー] 画面を拡大すると、プログラムのUI画像が適切に拡大されません。
- [エラー] 言語を変更した後、出力レポートの言語が変更されません。
- [エラー] 小数点を含むメッセージを誤ってURLとして認識する可能性があります。

[すべてのリリース](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/blob/main/assets/docs/Changelog.md)

## クイックスタート
**太字の項目は必須の要件です。**
### システム要件
- システム要件: 64ビットWindows
- **プロセッサ**: 64ビットプロセッサ
- **メモリ**: 2GB
- グラフィックスカード: 1GBのVRAMを搭載し、CUDAアクセラレーションをサポートするNVIDIAグラフィックスカード
- **ストレージ**: 利用可能なスペース3GB

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
- Python
```shell
pip install Pillow
pip install requests
pip install numpy
```

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


## GUI
![t2i](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/blob/main/assets/samples/GUI.png)
### GUIを起動
```shell
python Nagato-Sakura-SMS-Checker-GUI.py
```

### 言語
現在、GUIは繁体字中国語、英語、日本語の迅速な切り替えをサポートしています。切り替えには、上部のメニューバー/言語を使用してください。

### テーマ
現在、GUIには現代的な明るいテーマ、現代的な暗いテーマ、紅の翼、青い影、ダークインディゴ、かわいいディメンション、そしてディメンションの再生の7つのテーマがあります。上部のメニューバー/テーマを使用して切り替えてください。
![t2i](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/blob/main/assets/samples/themes/Cute_Dimension.png)


## GUIの実用例
### 例1
SMSの内容：Myfoneリマインダー：2月29日時点で、あなたの番号には19,985ポイントが残っています。本日有効期限が切れます。リンクをクリックしてプライズを受け取ってください！

この種のSMSは台湾で人気のある詐欺SMSです。メッセージだけでなく、詐欺のウェブサイトも非常に説得力があります。長門さくらの識別結果を見てみましょう：

![t2i](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/blob/main/assets/samples/example/scam_sms.png)

長門さくらは詐欺メッセージを正常に識別し、SMS内のURLを基本的に検査します。ここでは、URLがHTTPSではなくHTTPを使用しているため、長門さくらはユーザーに潜在的なリスクを警告するメッセージを発行します。


### 例2
SMS の内容：Hami書店の月間読書パッケージ「期間限定ダウンロード」は、1年で360冊以上の書籍を提供します！メンバーはすぐにお気に入りの本に投票することができ、hamibook.tw/2NQYpで500元を獲得するチャンスがあります。

この種のSMSは、台湾でよく見られる電気通信会社の広告です。中華電信と亜太電信の両方が受け取ります。 このようなメッセージのURLは、時々SMS内に特別な短縮URLであることがあります。この難しいタスクを長門サクラが処理できるでしょうか？

![t2i](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/blob/main/assets/samples/example/advertise_sms.png)

長門サクラは、広告メッセージを正常に識別し、httpやwwwで始まるURLを検出できなかったURLを正常に検出し、それらを正しいURLに変換し、それらをテストしました。この広告は問題なく安全であるようです。

### 例3
SMS の内容：OPEN POINT会員の皆様、確認コードは47385です。これが本人によるものでない場合は、すぐにパスワードを変更することをお勧めします。注意！パスワードや確認コードを他人に漏らさないでください。

この種のSMSは一般的な確認コードのSMSです。長門サクラはこれをどのように処理するのか見てみましょう。

![t2i](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/blob/main/assets/samples/example/captcha_sms.png)

長門サクラは、確認コードのSMSを正常に識別しました。ただし、バージョン1.0.2では、AppleのようにSMSから確認コードを抽出することができません。長門サクラはまだ頑張る必要があるようです。

### 例4
SMS の内容：2023/11/24 14:19 あなたは0918001824からの未着信1通があります。重要な電話に返信することをお忘れなく！ 既に通話を受けたり返信した場合は、このSMSを無視してください。

この種のSMSは、ほとんどの人が受け取る未着信のメッセージです。長門サクラはこれをどのように処理するのでしょうか？

![t2i](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/blob/main/assets/samples/example/normal_sms.png)

長門サクラは、未着信のメッセージを一般的なメッセージとして分類し、メッセージから正しく電話番号を読み取りました。本当に素晴らしいです。長門サクラに拍手を送りましょう。

### 例 EX
SMS の内容：【亜太電信請求通知】現在の請求額は349元です。この請求書は次の期間に統合されて送信されます。請求書のお問い合わせやオンライン支払いには、当社のモバイル顧客サービスAPP、公式ウェブサイトの会員エリアをご利用ください。また、携帯電話で直接988にダイヤルしたり、7-11 ibonで支払いをすることもできます。支払いが完了した場合は、この通知を無視してください。ありがとうございます。

長門サクラは、私たちがやったことは自分を過小評価していると言います。彼女に2つ以上のURLを与えてみるように私たちに頼みます。長門サクラ、無理をしないでください。

![t2i](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/blob/main/assets/samples/example/two_url_sms.png)

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
以下のプロジェクトと貢献者に特に感謝します：

### プロジェクト
- [requests](https://github.com/psf/requests)
- [165反詐騙諮詢專線_假投資(博弈)網站](https://data.gov.tw/dataset/160055)

### 貢献者
<a href="https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=AmanoShizukikun/Nagato-Sakura-SMS-Checker" />
</a>
