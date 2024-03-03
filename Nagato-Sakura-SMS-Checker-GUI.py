import tkinter as tk
from tkinter import messagebox, scrolledtext, Menu, filedialog
import torch
import torch.nn as nn
import json
from pathlib import Path
from PIL import Image, ImageTk
import re
import requests
import ssl
import socket
import threading
from urllib.parse import urlparse
import webbrowser

# 檢查是否有可用的 NVIDIA 顯示卡並設置為運算裝置、檢測環境
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("<環境檢測>")
print(torch.__version__)
print(device)

# 定義全域變量
current_directory = Path(__file__).resolve().parent

# 定義神經網路模型
class SMSClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SMSClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, int(hidden_size * 0.66)) 
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(int(hidden_size * 0.66), int(hidden_size * 0.66)) 
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(int(hidden_size * 0.66), output_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.fc1(x)  # 輸入層到第一個隱藏層
        x = self.relu1(x)
        x = self.fc2(x)  # 第一個隱藏層到第二個隱藏層
        x = self.relu2(x)
        x = self.fc3(x)  # 最後一個隱藏層到輸出層
        x = self.softmax(x)
        return x

# 加載模型和 vocab
def load_model(model_path, vocab_path, config_path, label_path, device):
    with open(vocab_path, 'r') as json_file:
        vocab = json.load(json_file)
    with open(config_path, 'r') as json_file:
        model_config = json.load(json_file)
    with open(label_path, 'r') as labels_file:
        label_mapping = {}
        for line in labels_file:
            label, index = line.strip().split(': ')
            label_mapping[label] = int(index)
    model = SMSClassifier(model_config['input_size'], model_config['hidden_size'], model_config['output_size'])
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    return model, vocab, label_mapping

# 設定檔案路徑並載入模型、詞彙表和標籤映射到裝置
VOCAB_PATH = current_directory / "models" / "tokenizer.json"
MODEL_PATH = current_directory / "models" / "SMS_model.bin"
CONFIG_PATH = current_directory / "models" / "config.json"
LABEL_PATH = current_directory / "models" / "labels.txt"
BLACKLIST_PATH = current_directory / "data" / "blacklist.txt"
model, vocab, label_mapping = load_model(MODEL_PATH, VOCAB_PATH, CONFIG_PATH, LABEL_PATH, device)

# 將文本轉換為向量
def text_to_vector(text):
    vector = [0] * len(vocab)
    for word in text:
        if word in vocab:
            vector[vocab.index(word)] = 1
    return vector

# 獲取使用者輸入的簡訊內容
def predict_and_display():
    language = current_language
    user_input = entry.get()
    if user_input == "":
        if language == "繁體中文":
            messagebox.showinfo("提醒", "請輸入簡訊內容！")
        elif language == "English":
            messagebox.showinfo("Reminder", "Please enter the SMS content!")
        elif language == "日本語":
            messagebox.showinfo("リマインダー", "メッセージ内容を入力してください！")
    else:
        predict_SMS(user_input, safety_text_box)
        
# 預測簡訊類別並顯示結果
def predict_SMS(text, text_widget):
    language = current_language
    input_vector = text_to_vector(text)
    input_vector = torch.tensor(input_vector, dtype=torch.float).unsqueeze(0).to(device)
    output = model(input_vector)
    predicted_class = torch.argmax(output).item()
    predicted_probs = output.squeeze().tolist()
    predicted_label = [label for label, index in label_mapping.items() if index == predicted_class][0]
    phone_numbers = re.findall(r'(\(?0\d{1,2}\)?[-\.\s]?\d{3,4}[-\.\s]?\d{3,4})', text)
    urls = re.findall(r'\b(?:https?://)?(?:www\.)?[\w\.-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?(?![\w\.-])\b', text)
    verification_codes = re.findall(r'(?<!\d)(\d{4,6})(?!\d)(?<!/)', text)
    if language == "繁體中文":
        print(f"【簡訊內容】:{text}")
        text_widget.insert(tk.END, f"【簡訊內容】:{text}\n")
        print(f"【預測概率】: {predicted_probs}")
        text_widget.insert(tk.END, f"【預測概率】: {predicted_probs}\n")
        print(f"【預測結果】: {predicted_label}")
        text_widget.insert(tk.END, f"【預測結果】: {predicted_label}\n")
        if phone_numbers:
            print(f"【偵測電話】:{phone_numbers}")
            text_widget.insert(tk.END, f"【偵測電話】:{phone_numbers}\n")
        if urls:
            for url in urls:
                if current_blacklist == "開啟":
                    with open(BLACKLIST_PATH, "r", encoding="utf-8") as file:
                        blacklist = set(line.strip() for line in file)
                    for blacklisted_url in blacklist:
                        if blacklisted_url in urls:
                            print(f"【黑名單】 {urls} 在黑名單中")
                            text_widget.insert(tk.END, f"【黑名單】 {urls} 在黑名單中\n")
                            break  
                threading.Thread(target=check_url_safety, args=(url, text_widget)).start()
        if predicted_label == 'Captcha SMS':
            if verification_codes:
                print(f"【驗證碼】:{verification_codes}")
                text_widget.insert(tk.END, f"【驗證碼】:{verification_codes}\n")
            else:
                print("【驗證碼】:未找到驗證碼。")
                text_widget.insert(tk.END, "【驗證碼】:未找到驗證碼。\n")
    elif language == "English":
        print(f"【SMS Content】:{text}")
        text_widget.insert(tk.END, f"【SMS Content】:{text}\n")
        print(f"【Prediction Probabilities】: {predicted_probs}")
        text_widget.insert(tk.END, f"【Prediction Probabilities】: {predicted_probs}\n")
        print(f"【Prediction Result】: {predicted_label}")
        text_widget.insert(tk.END, f"【Prediction Result】: {predicted_label}\n")
        if phone_numbers:
            print(f"【Detected Phone Numbers】:{phone_numbers}")
            text_widget.insert(tk.END, f"【Detected Phone Numbers】:{phone_numbers}\n")
        if urls:
            for url in urls:
                if current_blacklist == "開啟":
                    with open(BLACKLIST_PATH, "r", encoding="utf-8") as file:
                        blacklist = set(line.strip() for line in file)
                    for blacklisted_url in blacklist:
                        if blacklisted_url in urls:
                            print(f"【Blacklist】 {urls} is in the blacklist\n")
                            text_widget.insert(tk.END, f"【Blacklist】 {urls} is in the blacklist\n")
                            break  
                threading.Thread(target=check_url_safety, args=(url, text_widget)).start()
        if predicted_label == 'Captcha SMS':
            if verification_codes:
                print(f"【Verification Codes】:{verification_codes}")
                text_widget.insert(tk.END, f"【Verification Codes】:{verification_codes}\n")
            else:
                print("【Verification Codes】:No verification code found.")
                text_widget.insert(tk.END, "【Verification Codes】:No verification code found.\n")
    elif language == "日本語":
        print(f"【SMS コンテンツ】:{text}")
        text_widget.insert(tk.END, f"【SMS コンテンツ】:{text}\n")
        print(f"【予測確率】: {predicted_probs}")
        text_widget.insert(tk.END, f"【予測確率】: {predicted_probs}\n")
        print(f"【予測結果】: {predicted_label}")
        text_widget.insert(tk.END, f"【予測結果】: {predicted_label}\n")
        if phone_numbers:
            print(f"【検出電話】:{phone_numbers}")
            text_widget.insert(tk.END, f"【検出電話】:{phone_numbers}\n")
        if urls:
            for url in urls:
                if current_blacklist == "開啟":
                    with open(BLACKLIST_PATH, "r", encoding="utf-8") as file:
                        blacklist = set(line.strip() for line in file)
                    for blacklisted_url in blacklist:
                        if blacklisted_url in urls:
                            print(f"【ブラックリスト】 {urls} はブラックリストにあります\n")
                            text_widget.insert(tk.END, f"【ブラックリスト】 {urls} はブラックリストにあります\n")
                            break  
                threading.Thread(target=check_url_safety, args=(url, text_widget)).start()
        if predicted_label == 'Captcha SMS':
            if verification_codes:
                print(f"【確認コード】:{verification_codes}")
                text_widget.insert(tk.END, f"【確認コード】:{verification_codes}\n")
            else:
                print("【確認コード】:確認コードが見つかりませんでした。")
                text_widget.insert(tk.END, "【確認コード】:確認コードが見つかりませんでした。\n")

    return predicted_label, predicted_probs, predicted_class, phone_numbers, urls, verification_codes

# 檢查網址安全性
def check_url_safety(url, text_widget):
    language = current_language
    show_hostname = current_hostname
    show_cert = current_cert
    try:
        # 若網址不是以 http:// 或 https:// 開頭，則加上 https://
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        
        # 若網址不是以 https:// 開頭，則印出警告訊息
        if not url.startswith("https://"):
            if language == "繁體中文":
                result = f"【警告】 {url} 使用不安全的協議\n"
            elif language == "English":
                result = f"【Warning】 {url} is using an insecure protocol\n"
            elif language == "日本語":
                result = f"【警告】 {url} は安全でないプロトコルを使用しています\n"
            else:
                result = f"【警告】Unknown language: {language}\n"
            print(result)
            text_widget.insert(tk.END, result)
        
        # 檢查網址中是否含有可疑模式
        suspicious_patterns = ["phishing", "malware", "hack", "top"]
        if any(pattern in url.lower() for pattern in suspicious_patterns):
            if language == "繁體中文":
                result = f"【警告】 {url} 的路徑包含可疑模式\n"
            elif language == "English":
                result = f"【Warning】 The path of {url} contains suspicious patterns\n"
            elif language == "日本語":
                result = f"【警告】 {url} のパスには疑わしいパターンが含まれています\n"
            else:
                result = f"【警告】Unknown language: {language}\n"
            print(result)
            text_widget.insert(tk.END, result)
        
        # 解析網址並取得主機名稱和 IP 地址
        parsed_url = urlparse(url)
        hostname = parsed_url.hostname
        ip_address = socket.gethostbyname(hostname)
        hostname_from_ip = socket.gethostbyaddr(ip_address)
        if show_hostname == "開啟":
            if language == "繁體中文":
                print(f"【主機名稱】{hostname}")
                text_widget.insert(tk.END, f"【主機名稱】{hostname}\n")
                print(f"【IP 位址】{ip_address}")
                text_widget.insert(tk.END, f"【IP 位址】{ip_address}\n")
                print(f"【從 IP 位址取得的主機名稱】{hostname_from_ip}")
                text_widget.insert(tk.END, f"【從 IP 位址取得的主機名稱】{hostname_from_ip}\n")
            elif language == "English":
                print(f"【Hostname】: {hostname}")
                text_widget.insert(tk.END, f"【Hostname】: {hostname}\n")
                print(f"【IP Address】: {ip_address}")
                text_widget.insert(tk.END, f"【IP Address】: {ip_address}\n")
                print(f"【Hostname from IP Address】: {hostname_from_ip}")
                text_widget.insert(tk.END, f"【Hostname from IP Address】: {hostname_from_ip}\n")
            elif language == "日本語":
                print(f"【ホスト名】: {hostname}")
                text_widget.insert(tk.END, f"【ホスト名】: {hostname}\n")
                print(f"【IPアドレス】: {ip_address}")
                text_widget.insert(tk.END, f"【IPアドレス】: {ip_address}\n")
                print(f"【IPアドレスからのホスト名】: {hostname_from_ip}")
                text_widget.insert(tk.END, f"【IPアドレスからのホスト名】: {hostname_from_ip}\n")
                
        # 建立 SSL 連線並取得伺服器憑證
        context = ssl.create_default_context()
        context.check_hostname = False
        with context.wrap_socket(socket.socket(), server_hostname=url) as s:
            s.settimeout(5)
            s.connect((hostname, 443))
            cert = s.getpeercert()
        cert_start_date = cert['notBefore']
        cert_end_date = cert['notAfter']
        if show_cert == "開啟":
            if language == "繁體中文":
                print(f"【憑證起始日期】：{cert_start_date}")
                text_widget.insert(tk.END, f"【憑證起始日期】：{cert_start_date}\n")
                print(f"【憑證結束日期】：{cert_end_date}")
                text_widget.insert(tk.END, f"【憑證結束日期】：{cert_end_date}\n")
            elif language == "English":
                print(f"【Certificate Start Date】：{cert_start_date}")
                text_widget.insert(tk.END, f"【Certificate Start Date】：{cert_start_date}\n")
                print(f"【Certificate End Date】：{cert_end_date}")
                text_widget.insert(tk.END, f"【Certificate End Date】：{cert_end_date}\n")
            elif language == "日本語":
                print(f"【証明書開始日】：{cert_start_date}")
                text_widget.insert(tk.END, f"【証明書開始日】：{cert_start_date}\n")
                print(f"【証明書終了日】：{cert_end_date}")
                text_widget.insert(tk.END, f"【証明書終了日】：{cert_end_date}\n")
        
        # 發送 HTTP 請求並檢查回應狀態碼
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            if language == "繁體中文":
                result = f"【安全】 {url} 是安全的\n"
            elif language == "English":
                result = f"【Safe】 {url} is safe\n"
            elif language == "日本語":
                result = f"【安全】 {url} は安全です\n"
            else:
                result = f"【安全】Unknown language\: {language}\n"
            print(result)
            text_widget.insert(tk.END, result)
        else:
            if language == "繁體中文":
                result = f"【警告】 {url} 可能有風險 (狀態碼: {response.status_code}).\n"
            elif language == "English":
                result = f"【Warning】 {url} may be at risk (status code: {response.status_code}).\n"
            elif language == "日本語":
                result = f"【警告】 {url} はリスクがある可能性があります（ステータスコード: {response.status_code}）。\n"
            else:
                result = f"【警告】Unknown language: {language}\n"
            print(result)
            text_widget.insert(tk.END, result)

    except requests.exceptions.RequestException as e:
        if language == "繁體中文":
            result = f"【錯誤】 {url}: 請求錯誤 ({str(e)})\n"
        elif language == "English":
            result = f"【Error】 {url}: Request error ({str(e)})\n"
        elif language == "日本語":
            result = f"【エラー】 {url}: リクエストエラー ({str(e)})\n"
        else:
            result = f"【Error】Unknown language: {language}\n"
        print(result)
        text_widget.insert(tk.END, result)
        
    except ssl.SSLError as ssl_error:
        if language == "繁體中文":
            result = f"【警告】 {url} SSL 握手失敗 ({ssl_error.strerror})\n"
        elif language == "English":
            result = f"【Warning】 {url} SSL handshake failed ({ssl_error.strerror})\n"
        elif language == "日本語":
            result = f"【警告】 {url} SSL ハンドシェイクに失敗しました ({ssl_error.strerror})\n"
        else:
            result = f"【警告】Unknown language: {language}\n"
        print(result)
        text_widget.insert(tk.END, result)
        
    except socket.timeout:
        if language == "繁體中文":
            result = f"【錯誤】 {url}: 連接超時\n"
        elif language == "English":
            result = f"【Error】 {url}: Connection timeout\n"
        elif language == "日本語":
            result = f"【エラー】 {url}: 接続がタイムアウトしました\n"
        else:
            result = f"【Error】Unknown language: {language}\n"
        print(result)
        text_widget.insert(tk.END, result)
        
    except socket.error as socket_error:
        if language == "繁體中文":
            result = f"【錯誤】 {url}: 連接錯誤 ({str(socket_error)})\n"
        elif language == "English":
            result = f"【Error】 {url}: Connection error ({str(socket_error)})\n"
        elif language == "日本語":
            result = f"【エラー】 {url}: 接続エラー ({str(socket_error)})\n"
        else:
            result = f"【Error】Unknown language: {language}\n"
        print(result)
        text_widget.insert(tk.END, result)
        
    except Exception as e:
        if language == "繁體中文":
            result = f"【錯誤】 {url}: {str(e)}\n"
        elif language == "English":
            result = f"【Error】 {url}: {str(e)}\n"
        elif language == "日本語":
            result = f"【エラー】 {url}: {str(e)}\n"
        else:
            result = f"【Error】Unknown language: {language}\n"
        print(result)
        text_widget.insert(tk.END, result)
        
# 清除鍵        
def clear_input():
    entry.delete(0, tk.END)
    safety_text_box.delete('1.0', tk.END)

# 開啟檔案
def open_json_file():
    file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
    if file_path:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            safety_text_box.delete('1.0', tk.END)
            safety_text_box.insert(tk.END, content)

# 儲存檔案
def save_to_json():
    save_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as file:
            content = safety_text_box.get('1.0', tk.END)
            file.write(content)

# 語言選單
def set_language(language, theme, blacklist, show_hostname, show_cert):
    global current_language
    current_language = language
    global current_theme
    current_theme = theme
    global current_blacklist
    current_blacklist = blacklist
    global current_hostname
    current_hostname = show_hostname
    global current_cert
    current_cert = show_cert
    global version
    global predict_button
    global clear_button
    global file_menu
    global language_menu
    global theme_menu
    global root
    global menu_bar
    global help_menu
    
    if language == "繁體中文":
        predict_button.config(text="預測")
        clear_button.config(text="清除")
        root.config(menu=None) 
        menu_bar = Menu(root)
        root.config(menu=menu_bar)

        file_menu = Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="檔案", menu=file_menu)
        file_menu.add_command(label="開啟", command=open_json_file)
        file_menu.add_command(label="儲存", command=save_to_json)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=root.quit)

        language_menu = Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="語言", menu=language_menu)
        language_menu.add_command(label="☑ 繁體中文", command=lambda: set_language("繁體中文", current_theme, current_blacklist, current_hostname, current_cert))
        language_menu.add_command(label="☐ English", command=lambda: set_language("English", current_theme, current_blacklist, current_hostname, current_cert))
        language_menu.add_command(label="☐ 日本語", command=lambda: set_language("日本語", current_theme, current_blacklist, current_hostname, current_cert))
        
        help_menu = Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="說明", menu=help_menu)
        help_menu.add_command(label="開啟網站", command=open_github)
        help_menu.add_separator()
        help_menu.add_command(label="版本資訊", command=show_version_info)

        theme_menu = Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="主題", menu=theme_menu)
        if theme == "現代淺色":
            theme_menu.add_command(label="☑ 現代淺色", command=lambda: set_theme("現代淺色"))
            theme_menu.add_command(label="☐ 現代深色", command=lambda: set_theme("現代深色"))
            theme_menu.add_command(label="☐ 緋紅之翼", command=lambda: set_theme("緋紅之翼"))
            theme_menu.add_command(label="☐ 青色陰影", command=lambda: set_theme("青色陰影"))
            theme_menu.add_command(label="☐ 暗度藏青", command=lambda: set_theme("暗度藏青"))
            theme_menu.add_command(label="☐ 可愛次元", command=lambda: set_theme("可愛次元"))
            theme_menu.add_command(label="☐ 次元重生", command=lambda: set_theme("次元重生"))
            theme_menu.add_command(label="☐ 賽博朋克", command=lambda: set_theme("賽博朋克"))
        elif theme == "現代深色":
            theme_menu.add_command(label="☐ 現代淺色", command=lambda: set_theme("現代淺色"))
            theme_menu.add_command(label="☑ 現代深色", command=lambda: set_theme("現代深色"))
            theme_menu.add_command(label="☐ 緋紅之翼", command=lambda: set_theme("緋紅之翼"))
            theme_menu.add_command(label="☐ 青色陰影", command=lambda: set_theme("青色陰影"))
            theme_menu.add_command(label="☐ 暗度藏青", command=lambda: set_theme("暗度藏青"))
            theme_menu.add_command(label="☐ 可愛次元", command=lambda: set_theme("可愛次元"))
            theme_menu.add_command(label="☐ 次元重生", command=lambda: set_theme("次元重生"))
            theme_menu.add_command(label="☐ 賽博朋克", command=lambda: set_theme("賽博朋克"))
        elif theme == "緋紅之翼":
            theme_menu.add_command(label="☐ 現代淺色", command=lambda: set_theme("現代淺色"))
            theme_menu.add_command(label="☐ 現代深色", command=lambda: set_theme("現代深色"))
            theme_menu.add_command(label="☑ 緋紅之翼", command=lambda: set_theme("緋紅之翼"))
            theme_menu.add_command(label="☐ 青色陰影", command=lambda: set_theme("青色陰影"))
            theme_menu.add_command(label="☐ 暗度藏青", command=lambda: set_theme("暗度藏青"))
            theme_menu.add_command(label="☐ 可愛次元", command=lambda: set_theme("可愛次元"))
            theme_menu.add_command(label="☐ 次元重生", command=lambda: set_theme("次元重生"))
            theme_menu.add_command(label="☐ 賽博朋克", command=lambda: set_theme("賽博朋克"))
        elif theme == "青色陰影":
            theme_menu.add_command(label="☐ 現代淺色", command=lambda: set_theme("現代淺色"))
            theme_menu.add_command(label="☐ 現代深色", command=lambda: set_theme("現代深色"))
            theme_menu.add_command(label="☐ 緋紅之翼", command=lambda: set_theme("緋紅之翼"))
            theme_menu.add_command(label="☑ 青色陰影", command=lambda: set_theme("青色陰影"))
            theme_menu.add_command(label="☐ 暗度藏青", command=lambda: set_theme("暗度藏青"))
            theme_menu.add_command(label="☐ 可愛次元", command=lambda: set_theme("可愛次元"))
            theme_menu.add_command(label="☐ 次元重生", command=lambda: set_theme("次元重生"))
            theme_menu.add_command(label="☐ 賽博朋克", command=lambda: set_theme("賽博朋克"))
        elif theme == "暗度藏青":
            theme_menu.add_command(label="☐ 現代淺色", command=lambda: set_theme("現代淺色"))
            theme_menu.add_command(label="☐ 現代深色", command=lambda: set_theme("現代深色"))
            theme_menu.add_command(label="☐ 緋紅之翼", command=lambda: set_theme("緋紅之翼"))
            theme_menu.add_command(label="☐ 青色陰影", command=lambda: set_theme("青色陰影"))
            theme_menu.add_command(label="☑ 暗度藏青", command=lambda: set_theme("暗度藏青"))
            theme_menu.add_command(label="☐ 可愛次元", command=lambda: set_theme("可愛次元"))
            theme_menu.add_command(label="☐ 次元重生", command=lambda: set_theme("次元重生"))
            theme_menu.add_command(label="☐ 賽博朋克", command=lambda: set_theme("賽博朋克"))
        elif theme == "可愛次元":
            theme_menu.add_command(label="☐ 現代淺色", command=lambda: set_theme("現代淺色"))
            theme_menu.add_command(label="☐ 現代深色", command=lambda: set_theme("現代深色"))
            theme_menu.add_command(label="☐ 緋紅之翼", command=lambda: set_theme("緋紅之翼"))
            theme_menu.add_command(label="☐ 青色陰影", command=lambda: set_theme("青色陰影"))
            theme_menu.add_command(label="☐ 暗度藏青", command=lambda: set_theme("暗度藏青"))
            theme_menu.add_command(label="☑ 可愛次元", command=lambda: set_theme("可愛次元"))
            theme_menu.add_command(label="☐ 次元重生", command=lambda: set_theme("次元重生"))
            theme_menu.add_command(label="☐ 賽博朋克", command=lambda: set_theme("賽博朋克"))
        elif theme == "次元重生":
            theme_menu.add_command(label="☐ 現代淺色", command=lambda: set_theme("現代淺色"))
            theme_menu.add_command(label="☐ 現代深色", command=lambda: set_theme("現代深色"))
            theme_menu.add_command(label="☐ 緋紅之翼", command=lambda: set_theme("緋紅之翼"))
            theme_menu.add_command(label="☐ 青色陰影", command=lambda: set_theme("青色陰影"))
            theme_menu.add_command(label="☐ 暗度藏青", command=lambda: set_theme("暗度藏青"))
            theme_menu.add_command(label="☐ 可愛次元", command=lambda: set_theme("可愛次元"))
            theme_menu.add_command(label="☑ 次元重生", command=lambda: set_theme("次元重生"))
            theme_menu.add_command(label="☐ 賽博朋克", command=lambda: set_theme("賽博朋克"))
        elif theme == "賽博朋克":
            theme_menu.add_command(label="☐ 現代淺色", command=lambda: set_theme("現代淺色"))
            theme_menu.add_command(label="☐ 現代深色", command=lambda: set_theme("現代深色"))
            theme_menu.add_command(label="☐ 緋紅之翼", command=lambda: set_theme("緋紅之翼"))
            theme_menu.add_command(label="☐ 青色陰影", command=lambda: set_theme("青色陰影"))
            theme_menu.add_command(label="☐ 暗度藏青", command=lambda: set_theme("暗度藏青"))
            theme_menu.add_command(label="☐ 可愛次元", command=lambda: set_theme("可愛次元"))
            theme_menu.add_command(label="☐ 次元重生", command=lambda: set_theme("次元重生"))
            theme_menu.add_command(label="☑ 賽博朋克", command=lambda: set_theme("賽博朋克"))
            
        setting_menu = Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="設定", menu=setting_menu)
        if blacklist == "開啟":
            setting_menu.add_command(label="☑ 黑名單", command=lambda: enable_blacklist("關閉"))
        elif blacklist == "關閉":
            setting_menu.add_command(label="☐ 黑名單", command=lambda: enable_blacklist("開啟"))
        if show_hostname == "開啟":
            setting_menu.add_command(label="☑ 顯示主機名稱及IP", command=lambda: enable_hostname("關閉"))
        elif show_hostname == "關閉":
            setting_menu.add_command(label="☐ 顯示主機名稱及IP", command=lambda: enable_hostname("開啟"))
        if show_cert == "開啟":
            setting_menu.add_command(label="☑ 顯示憑證日期", command=lambda: enable_cert("關閉"))
        elif show_cert == "關閉":
            setting_menu.add_command(label="☐ 顯示憑證日期", command=lambda: enable_cert("開啟"))
        
    elif language == "English":
        predict_button.config(text="Predict")
        clear_button.config(text="Clear")
        root.config(menu=None)
        menu_bar = Menu(root)
        root.config(menu=menu_bar)

        file_menu = Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open", command=open_json_file)
        file_menu.add_command(label="Save", command=save_to_json)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=root.quit)

        language_menu = Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Language", menu=language_menu)
        language_menu.add_command(label="☐ 繁體中文", command=lambda: set_language("繁體中文", current_theme, current_blacklist, current_hostname, current_cert))
        language_menu.add_command(label="☑ English", command=lambda: set_language("English", current_theme, current_blacklist, current_hostname, current_cert))
        language_menu.add_command(label="☐ 日本語", command=lambda: set_language("日本語", current_theme, current_blacklist, current_hostname, current_cert))
        
        help_menu = Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Open Website", command=open_github)
        help_menu.add_separator()
        help_menu.add_command(label="Version Information", command=show_version_info)

        theme_menu = Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Theme", menu=theme_menu)
        if theme == "現代淺色":
            theme_menu.add_command(label="☑ Light Mode", command=lambda: set_theme("現代淺色"))
            theme_menu.add_command(label="☐ Dark Mode", command=lambda: set_theme("現代深色"))
            theme_menu.add_command(label="☐ Crimson Wings", command=lambda: set_theme("緋紅之翼"))
            theme_menu.add_command(label="☐ Azure Shadow", command=lambda: set_theme("青色陰影"))
            theme_menu.add_command(label="☐ Midnight Indigo", command=lambda: set_theme("暗度藏青"))
            theme_menu.add_command(label="☐ Cute Dimension", command=lambda: set_theme("可愛次元"))
            theme_menu.add_command(label="☐ Dimension Rebirth", command=lambda: set_theme("次元重生"))
            theme_menu.add_command(label="☐ Cyberpunk", command=lambda: set_theme("賽博朋克"))
        elif theme == "現代深色":
            theme_menu.add_command(label="☐ Light Mode", command=lambda: set_theme("現代淺色"))
            theme_menu.add_command(label="☑ Dark Mode", command=lambda: set_theme("現代深色"))
            theme_menu.add_command(label="☐ Crimson Wings", command=lambda: set_theme("緋紅之翼"))
            theme_menu.add_command(label="☐ Azure Shadow", command=lambda: set_theme("青色陰影"))
            theme_menu.add_command(label="☐ Midnight Indigo", command=lambda: set_theme("暗度藏青"))
            theme_menu.add_command(label="☐ Cute Dimension", command=lambda: set_theme("可愛次元"))
            theme_menu.add_command(label="☐ Dimension Rebirth", command=lambda: set_theme("次元重生"))
            theme_menu.add_command(label="☐ Cyberpunk", command=lambda: set_theme("賽博朋克"))
        elif theme == "緋紅之翼":
            theme_menu.add_command(label="☐ Light Mode", command=lambda: set_theme("現代淺色"))
            theme_menu.add_command(label="☐ Dark Mode", command=lambda: set_theme("現代深色"))
            theme_menu.add_command(label="☑ Crimson Wings", command=lambda: set_theme("緋紅之翼"))
            theme_menu.add_command(label="☐ Azure Shadow", command=lambda: set_theme("青色陰影"))
            theme_menu.add_command(label="☐ Midnight Indigo", command=lambda: set_theme("暗度藏青"))
            theme_menu.add_command(label="☐ Cute Dimension", command=lambda: set_theme("可愛次元"))
            theme_menu.add_command(label="☐ Dimension Rebirth", command=lambda: set_theme("次元重生"))
            theme_menu.add_command(label="☐ Cyberpunk", command=lambda: set_theme("賽博朋克"))
        elif theme == "青色陰影":
            theme_menu.add_command(label="☐ Light Mode", command=lambda: set_theme("現代淺色"))
            theme_menu.add_command(label="☐ Dark Mode", command=lambda: set_theme("現代深色"))
            theme_menu.add_command(label="☐ Crimson Wings", command=lambda: set_theme("緋紅之翼"))
            theme_menu.add_command(label="☑ Azure Shadow", command=lambda: set_theme("青色陰影"))
            theme_menu.add_command(label="☐ Midnight Indigo", command=lambda: set_theme("暗度藏青"))
            theme_menu.add_command(label="☐ Cute Dimension", command=lambda: set_theme("可愛次元"))
            theme_menu.add_command(label="☐ Dimension Rebirth", command=lambda: set_theme("次元重生"))
            theme_menu.add_command(label="☐ Cyberpunk", command=lambda: set_theme("賽博朋克"))
        elif theme == "暗度藏青":
            theme_menu.add_command(label="☐ Light Mode", command=lambda: set_theme("現代淺色"))
            theme_menu.add_command(label="☐ Dark Mode", command=lambda: set_theme("現代深色"))
            theme_menu.add_command(label="☐ Crimson Wings", command=lambda: set_theme("緋紅之翼"))
            theme_menu.add_command(label="☐ Azure Shadow", command=lambda: set_theme("青色陰影"))
            theme_menu.add_command(label="☑ Midnight Indigo", command=lambda: set_theme("暗度藏青"))
            theme_menu.add_command(label="☐ Cute Dimension", command=lambda: set_theme("可愛次元"))
            theme_menu.add_command(label="☐ Dimension Rebirth", command=lambda: set_theme("次元重生"))
            theme_menu.add_command(label="☐ Cyberpunk", command=lambda: set_theme("賽博朋克"))
        elif theme == "可愛次元":
            theme_menu.add_command(label="☐ Light Mode", command=lambda: set_theme("現代淺色"))
            theme_menu.add_command(label="☐ Dark Mode", command=lambda: set_theme("現代深色"))
            theme_menu.add_command(label="☐ Crimson Wings", command=lambda: set_theme("緋紅之翼"))
            theme_menu.add_command(label="☐ Azure Shadow", command=lambda: set_theme("青色陰影"))
            theme_menu.add_command(label="☐ Midnight Indigo", command=lambda: set_theme("暗度藏青"))
            theme_menu.add_command(label="☑ Cute Dimension", command=lambda: set_theme("可愛次元"))
            theme_menu.add_command(label="☐ Dimension Rebirth", command=lambda: set_theme("次元重生"))
            theme_menu.add_command(label="☐ Cyberpunk", command=lambda: set_theme("賽博朋克"))
        elif theme == "次元重生":
            theme_menu.add_command(label="☐ Light Mode", command=lambda: set_theme("現代淺色"))
            theme_menu.add_command(label="☐ Dark Mode", command=lambda: set_theme("現代深色"))
            theme_menu.add_command(label="☐ Crimson Wings", command=lambda: set_theme("緋紅之翼"))
            theme_menu.add_command(label="☐ Azure Shadow", command=lambda: set_theme("青色陰影"))
            theme_menu.add_command(label="☐ Midnight Indigo", command=lambda: set_theme("暗度藏青"))
            theme_menu.add_command(label="☐ Cute Dimension", command=lambda: set_theme("可愛次元"))
            theme_menu.add_command(label="☑ Dimension Rebirth", command=lambda: set_theme("次元重生"))
            theme_menu.add_command(label="☐ Cyberpunk", command=lambda: set_theme("賽博朋克"))
        elif theme == "賽博朋克":
            theme_menu.add_command(label="☐ Light Mode", command=lambda: set_theme("現代淺色"))
            theme_menu.add_command(label="☐ Dark Mode", command=lambda: set_theme("現代深色"))
            theme_menu.add_command(label="☐ Crimson Wings", command=lambda: set_theme("緋紅之翼"))
            theme_menu.add_command(label="☐ Azure Shadow", command=lambda: set_theme("青色陰影"))
            theme_menu.add_command(label="☐ Midnight Indigo", command=lambda: set_theme("暗度藏青"))
            theme_menu.add_command(label="☐ Cute Dimension", command=lambda: set_theme("可愛次元"))
            theme_menu.add_command(label="☐ Dimension Rebirth", command=lambda: set_theme("次元重生"))
            theme_menu.add_command(label="☑ Cyberpunk", command=lambda: set_theme("賽博朋克"))
            
        setting_menu = Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Settings", menu=setting_menu)
        if blacklist == "開啟":
            setting_menu.add_command(label="☑ Blacklist", command=lambda: enable_blacklist("關閉"))
        elif blacklist == "關閉":
            setting_menu.add_command(label="☐ Blacklist", command=lambda: enable_blacklist("開啟"))
        if show_hostname == "開啟":
            setting_menu.add_command(label="☑ Show Hostname and IP", command=lambda: enable_hostname("關閉"))
        elif show_hostname == "關閉":
            setting_menu.add_command(label="☐ Show Hostname and IP", command=lambda: enable_hostname("開啟"))
        if show_cert == "開啟":
            setting_menu.add_command(label="☑ Show Certificate Dates", command=lambda: enable_cert("關閉"))
        elif show_cert == "關閉":
            setting_menu.add_command(label="☐ Show Certificate Dates", command=lambda: enable_cert("開啟"))
    
    elif language == "日本語":
        predict_button.config(text="予測")
        clear_button.config(text="クリア")
        root.config(menu=None)
        menu_bar = Menu(root)
        root.config(menu=menu_bar)

        file_menu = Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="ファイル", menu=file_menu)
        file_menu.add_command(label="開く", command=open_json_file)
        file_menu.add_command(label="保存", command=save_to_json)
        file_menu.add_separator()
        file_menu.add_command(label="出口", command=root.quit)

        language_menu = Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="言語", menu=language_menu)
        language_menu.add_command(label="☐ 繁體中文", command=lambda: set_language("繁體中文", current_theme, current_blacklist, current_hostname, current_cert),)
        language_menu.add_command(label="☐ English", command=lambda: set_language("English", current_theme, current_blacklist, current_hostname, current_cert))
        language_menu.add_command(label="☑ 日本語", command=lambda: set_language("日本語", current_theme, current_blacklist, current_hostname, current_cert))
        
        help_menu = Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="ヘルプ", menu=help_menu)
        help_menu.add_command(label="ウェブサイトを開く", command=open_github)
        help_menu.add_separator()
        help_menu.add_command(label="バージョン情報", command=show_version_info)

        theme_menu = Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="テーマ", menu=theme_menu)
        if theme == "現代淺色":
            theme_menu.add_command(label="☑ ライトモード", command=lambda: set_theme("現代淺色"))
            theme_menu.add_command(label="☐ ダークモード", command=lambda: set_theme("現代深色"))
            theme_menu.add_command(label="☐ 緋紅の翼", command=lambda: set_theme("緋紅之翼"))
            theme_menu.add_command(label="☐ 蒼影", command=lambda: set_theme("青色陰影"))
            theme_menu.add_command(label="☐ 真夜中の藍", command=lambda: set_theme("暗度藏青"))
            theme_menu.add_command(label="☐ キュートディメンション", command=lambda: set_theme("可愛次元"))
            theme_menu.add_command(label="☐ ディメンションリバース", command=lambda: set_theme("次元重生"))
            theme_menu.add_command(label="☐ サイバーパンク", command=lambda: set_theme("賽博朋克"))
        elif theme == "現代深色":
            theme_menu.add_command(label="☐ ライトモード", command=lambda: set_theme("現代淺色"))
            theme_menu.add_command(label="☑ ダークモード", command=lambda: set_theme("現代深色"))
            theme_menu.add_command(label="☐ 緋紅の翼", command=lambda: set_theme("緋紅之翼"))
            theme_menu.add_command(label="☐ 蒼影", command=lambda: set_theme("青色陰影"))
            theme_menu.add_command(label="☐ 真夜中の藍", command=lambda: set_theme("暗度藏青"))
            theme_menu.add_command(label="☐ キュートディメンション", command=lambda: set_theme("可愛次元"))
            theme_menu.add_command(label="☐ ディメンションリバース", command=lambda: set_theme("次元重生"))
            theme_menu.add_command(label="☐ サイバーパンク", command=lambda: set_theme("賽博朋克"))
        elif theme == "緋紅之翼":
            theme_menu.add_command(label="☐ ライトモード", command=lambda: set_theme("現代淺色"))
            theme_menu.add_command(label="☐ ダークモード", command=lambda: set_theme("現代深色"))
            theme_menu.add_command(label="☑ 緋紅の翼", command=lambda: set_theme("緋紅之翼"))
            theme_menu.add_command(label="☐ 蒼影", command=lambda: set_theme("青色陰影"))
            theme_menu.add_command(label="☐ 真夜中の藍", command=lambda: set_theme("暗度藏青"))
            theme_menu.add_command(label="☐ キュートディメンション", command=lambda: set_theme("可愛次元"))
            theme_menu.add_command(label="☐ ディメンションリバース", command=lambda: set_theme("次元重生"))
            theme_menu.add_command(label="☐ サイバーパンク", command=lambda: set_theme("賽博朋克"))
        elif theme == "青色陰影":
            theme_menu.add_command(label="☐ ライトモード", command=lambda: set_theme("現代淺色"))
            theme_menu.add_command(label="☐ ダークモード", command=lambda: set_theme("現代深色"))
            theme_menu.add_command(label="☐ 緋紅の翼", command=lambda: set_theme("緋紅之翼"))
            theme_menu.add_command(label="☑ 蒼影", command=lambda: set_theme("青色陰影"))
            theme_menu.add_command(label="☐ 真夜中の藍", command=lambda: set_theme("暗度藏青"))
            theme_menu.add_command(label="☐ キュートディメンション", command=lambda: set_theme("可愛次元"))
            theme_menu.add_command(label="☐ ディメンションリバース", command=lambda: set_theme("次元重生"))
            theme_menu.add_command(label="☐ サイバーパンク", command=lambda: set_theme("賽博朋克"))
        elif theme == "暗度藏青":
            theme_menu.add_command(label="☐ ライトモード", command=lambda: set_theme("現代淺色"))
            theme_menu.add_command(label="☐ ダークモード", command=lambda: set_theme("現代深色"))
            theme_menu.add_command(label="☐ 緋紅の翼", command=lambda: set_theme("緋紅之翼"))
            theme_menu.add_command(label="☐ 蒼影", command=lambda: set_theme("青色陰影"))
            theme_menu.add_command(label="☑ 真夜中の藍", command=lambda: set_theme("暗度藏青"))
            theme_menu.add_command(label="☐ キュートディメンション", command=lambda: set_theme("可愛次元"))
            theme_menu.add_command(label="☐ ディメンションリバース", command=lambda: set_theme("次元重生"))
            theme_menu.add_command(label="☐ サイバーパンク", command=lambda: set_theme("賽博朋克"))
        elif theme == "可愛次元":
            theme_menu.add_command(label="☐ ライトモード", command=lambda: set_theme("現代淺色"))
            theme_menu.add_command(label="☐ ダークモード", command=lambda: set_theme("現代深色"))
            theme_menu.add_command(label="☐ 緋紅の翼", command=lambda: set_theme("緋紅之翼"))
            theme_menu.add_command(label="☐ 蒼影", command=lambda: set_theme("青色陰影"))
            theme_menu.add_command(label="☐ 真夜中の藍", command=lambda: set_theme("暗度藏青"))
            theme_menu.add_command(label="☑ キュートディメンション", command=lambda: set_theme("可愛次元"))
            theme_menu.add_command(label="☐ ディメンションリバース", command=lambda: set_theme("次元重生"))
            theme_menu.add_command(label="☐ サイバーパンク", command=lambda: set_theme("賽博朋克"))
        elif theme == "次元重生":
            theme_menu.add_command(label="☐ ライトモード", command=lambda: set_theme("現代淺色"))
            theme_menu.add_command(label="☐ ダークモード", command=lambda: set_theme("現代深色"))
            theme_menu.add_command(label="☐ 緋紅の翼", command=lambda: set_theme("緋紅之翼"))
            theme_menu.add_command(label="☐ 蒼影", command=lambda: set_theme("青色陰影"))
            theme_menu.add_command(label="☐ 真夜中の藍", command=lambda: set_theme("暗度藏青"))
            theme_menu.add_command(label="☐ キュートディメンション", command=lambda: set_theme("可愛次元"))
            theme_menu.add_command(label="☑ ディメンションリバース", command=lambda: set_theme("次元重生"))
            theme_menu.add_command(label="☐ サイバーパンク", command=lambda: set_theme("賽博朋克"))
        elif theme == "賽博朋克":
            theme_menu.add_command(label="☐ ライトモード", command=lambda: set_theme("現代淺色"))
            theme_menu.add_command(label="☐ ダークモード", command=lambda: set_theme("現代深色"))
            theme_menu.add_command(label="☐ 緋紅の翼", command=lambda: set_theme("緋紅之翼"))
            theme_menu.add_command(label="☐ 蒼影", command=lambda: set_theme("青色陰影"))
            theme_menu.add_command(label="☐ 真夜中の藍", command=lambda: set_theme("暗度藏青"))
            theme_menu.add_command(label="☐ キュートディメンション", command=lambda: set_theme("可愛次元"))
            theme_menu.add_command(label="☐ ディメンションリバース", command=lambda: set_theme("次元重生"))
            theme_menu.add_command(label="☑ サイバーパンク", command=lambda: set_theme("賽博朋克"))
            
        setting_menu = Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="設定", menu=setting_menu)
        if blacklist == "開啟":
            setting_menu.add_command(label="☑ ブラックリスト", command=lambda: enable_blacklist("關閉"))
        elif blacklist == "關閉":
            setting_menu.add_command(label="☐ ブラックリスト", command=lambda: enable_blacklist("開啟"))
        if show_hostname == "開啟":
            setting_menu.add_command(label="☑ ホスト名とIPを表示", command=lambda: enable_hostname("關閉"))
        elif show_hostname == "關閉":
            setting_menu.add_command(label="☐ ホスト名とIPを表示", command=lambda: enable_hostname("開啟"))
        if show_cert == "開啟":
            setting_menu.add_command(label="☑ 証明書の日付を表示", command=lambda: enable_cert("關閉"))
        elif show_cert == "關閉":
            setting_menu.add_command(label="☐ 証明書の日付を表示", command=lambda: enable_cert("開啟"))

# 開啟網站
def open_github():
    webbrowser.open("https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker")
    
# 版本資訊
def load_model_version(config_path):
    with open(config_path, 'r') as json_file:
        config_data = json.load(json_file)
        model_version = config_data.get("version", "Unknown")
    return model_version

def show_version_info():
    language = current_language
    model_version = load_model_version(CONFIG_PATH)
    if language == "繁體中文":
        messagebox.showinfo("版本資訊", f"運行模式: {device}\nGUI 版本: {version}\n模型版本: {model_version}")
    elif language == "English":
        messagebox.showinfo("Version Information", f"Operating Mode: {device}\nGUI Version: {version}\nModel Version: {model_version}")
    elif language == "日本語":
        messagebox.showinfo("バージョン情報", f"動作モード: {device}\nGUI バージョン: {version}\nモデルバージョン: {model_version}")
        
# 主題
def set_theme(theme):
    global current_theme
    current_theme = theme
    if theme == "現代深色":
        root.config(bg='#1E1E1E')
        image_label.config(bg='#1E1E1E')
        entry.config(bg='#1F1F1F', fg='white', insertbackground='white')
        button_frame.config(bg='#1E1E1E')
        safety_text_box.config(bg='#1F1F1F', fg='white')
        predict_button.config(bg='#1E1E1E', fg='white')
        clear_button.config(bg='#1E1E1E', fg='white')
        empty_label.config(bg='#1E1E1E')
        menu_bar.config(bg='#1E1E1E')
    elif theme == "現代淺色":
        root.config(bg='white')
        image_label.config(bg='white')
        entry.config(bg='white', fg='black')
        button_frame.config(bg='white')
        safety_text_box.config(bg='white', fg='black')
        predict_button.config(bg='white', fg='black')
        clear_button.config(bg='white', fg='black')
        empty_label.config(bg='white')
        menu_bar.config(bg='white') 
    elif theme == "緋紅之翼":
        root.config(bg='#330000')
        image_label.config(bg='#330000')
        entry.config(bg='#390000', fg='white', insertbackground='white')
        button_frame.config(bg='#330000')
        safety_text_box.config(bg='#390000', fg='white')
        predict_button.config(bg='#330000', fg='white')
        clear_button.config(bg='#330000', fg='white')
        empty_label.config(bg='#330000')
        menu_bar.config(bg='#330000')
    elif theme == "青色陰影":
        root.config(bg='#00212B')
        image_label.config(bg='#00212B')
        entry.config(bg='#002B36', fg='white', insertbackground='white')
        button_frame.config(bg='#00212B')
        safety_text_box.config(bg='#002B36', fg='white')
        predict_button.config(bg='#00212B', fg='white')
        clear_button.config(bg='#00212B', fg='white')
        empty_label.config(bg='#00212B')
        menu_bar.config(bg='#00212B')
    elif theme == "暗度藏青":
        root.config(bg='#001C40')
        image_label.config(bg='#001C40')
        entry.config(bg='#002451', fg='white', insertbackground='white')
        button_frame.config(bg='#001C40')
        safety_text_box.config(bg='#002451', fg='white')
        predict_button.config(bg='#001C40', fg='white')
        clear_button.config(bg='#001C40', fg='white')
        empty_label.config(bg='#001C40')
        menu_bar.config(bg='#001C40')
    elif theme == "可愛次元":
        root.config(bg='#FCC2FD')
        image_label.config(bg='#FCC2FD')
        entry.config(bg='#FCCFFE', fg='black', insertbackground='black')
        button_frame.config(bg='#FCC2FD')
        safety_text_box.config(bg='#FCCFFE', fg='black')
        predict_button.config(bg='#FCC2FD', fg='black')
        clear_button.config(bg='#FCC2FD', fg='black')
        empty_label.config(bg='#FCC2FD')
        menu_bar.config(bg='#FCC2FD')
    elif theme == "次元重生":
        root.config(bg='#B5EDEF')
        image_label.config(bg='#B5EDEF')
        entry.config(bg='#FDE5FA', fg='black', insertbackground='black')
        button_frame.config(bg='#B5EDEF')
        safety_text_box.config(bg='#FDE5FA', fg='black')
        predict_button.config(bg='#B5EDEF', fg='black')
        clear_button.config(bg='#B5EDEF', fg='black')
        empty_label.config(bg='#B5EDEF')
        menu_bar.config(bg='#B5EDEF')
    elif theme == "賽博朋克":
        root.config(bg='#F2EA65')
        image_label.config(bg='#F2EA65')
        entry.config(bg='#65D0FC', fg='black', insertbackground='black')
        button_frame.config(bg='#F2EA65')
        safety_text_box.config(bg='#65D0FC', fg='black')
        predict_button.config(bg='#F2EA65', fg='black')
        clear_button.config(bg='#F2EA65', fg='black')
        empty_label.config(bg='#F2EA65')
        menu_bar.config(bg='#F2EA65')
    set_language(current_language, current_theme, current_blacklist, current_hostname, current_cert)
    
# 黑名單 
def enable_blacklist(blacklist):
    global current_blacklist
    current_blacklist = blacklist
    set_language(current_language, current_theme, current_blacklist, current_hostname, current_cert)
    
# 顯示憑證日期
def enable_hostname(show_hostname):
    global current_hostname
    current_hostname = show_hostname
    set_language(current_language, current_theme, current_blacklist, current_hostname, current_cert)
    
# 顯示憑證日期
def enable_cert(show_cert):
    global current_cert
    current_cert = show_cert
    set_language(current_language, current_theme, current_blacklist, current_hostname, current_cert)
    
def paste_text():
    entry.event_generate("<<Paste>>")

# 建立 tkinter 視窗
version = "1.0.4"
root = tk.Tk()
root.title(f"Nagato-Sakura-SMS-Checker-GUI-Ver.{version}")
root.geometry("820x580")  
icon_path = current_directory / "assets" / "icon" / f"{version}.ico"
root.iconbitmap(icon_path)

# 圖片
image_path = current_directory / "assets" / "4K" / f"{version}.jpg"
img = Image.open(image_path)
img_width, img_height = img.size
window_width = 700
new_img_width = window_width-100
new_img_height = int(img_height * (new_img_width / img_width))
img = img.resize((new_img_width, new_img_height), Image.LANCZOS)
photo = ImageTk.PhotoImage(img)
image_label = tk.Label(root, image=photo)
image_label.pack(fill="both", expand=True, padx=30)

# 輸入框
entry = tk.Entry(root, width=60)
entry.pack(fill="x", expand=True, padx=60)

# 右鍵選單
right_click_menu = tk.Menu(root, tearoff=0)
right_click_menu.add_command(label="貼上", command=paste_text)

def on_right_click(event):
    right_click_menu.post(event.x_root, event.y_root)
entry.bind("<Button-3>", on_right_click)

# 預測和清除按鈕
button_frame = tk.Frame(root)
button_frame.pack()
predict_button = tk.Button(button_frame, text="預測", command=predict_and_display)
predict_button.pack(side="left", padx=5)
clear_button = tk.Button(button_frame, text="清除", command=clear_input)
clear_button.pack(side="left")
empty_label = tk.Label(root, text="")
empty_label.pack()

# 滾動文字框
safety_text_box = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=("Arial", 11))
safety_text_box.pack(fill="both", expand=True, padx=20, pady=15)

# 選單列
menu_bar = Menu(root)
root.config(menu=menu_bar)
set_language("繁體中文","現代淺色","開啟","關閉","關閉")

# GUI 迴圈
root.mainloop()
