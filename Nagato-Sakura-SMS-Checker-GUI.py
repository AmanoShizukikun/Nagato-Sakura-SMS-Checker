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

# 檢查是否有可用的 NVIDIA 顯示卡，並設置運算裝置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 環境檢測
print("<環境檢測>")
print(torch.__version__)
print(device)

# 取得目前執行程式的資料夾路徑
current_directory = Path(__file__).resolve().parent

# 定義神經網路模型
class SMSClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SMSClassifier, self).__init__()
        # 輸入層到第一個隱藏層
        self.fc1 = nn.Linear(input_size, int(hidden_size * 0.66)) 
        self.relu1 = nn.ReLU()
        # 第一個隱藏層到第二個隱藏層
        self.fc2 = nn.Linear(int(hidden_size * 0.66), int(hidden_size * 0.66)) 
        self.relu2 = nn.ReLU()
        # 最後一個隱藏層到輸出層
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

# 將文本轉換為向量
def text_to_vector(text):
    vector = [0] * len(vocab)
    for word in text:
        if word in vocab:
            vector[vocab.index(word)] = 1
    return vector

# 加載模型和 vocab
def load_model(model_path, vocab_path, config_path, label_path, device):
    # 讀取 vocab
    with open(vocab_path, 'r') as json_file:
        vocab = json.load(json_file)
    # 讀取模型配置
    with open(config_path, 'r') as json_file:
        model_config = json.load(json_file)
    # 讀取標籤映射
    with open(label_path, 'r') as labels_file:
        label_mapping = {}
        for line in labels_file:
            label, index = line.strip().split(': ')
            label_mapping[label] = int(index)
    # 初始化模型並將其移到 NVIDIA 顯示卡上
    model = SMSClassifier(model_config['input_size'], model_config['hidden_size'], model_config['output_size'])
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    return model, vocab, label_mapping

# 設定檔案路徑
VOCAB_PATH = current_directory / "models" / "tokenizer.json"
MODEL_PATH = current_directory / "models" / "SMS_model.bin"
CONFIG_PATH = current_directory / "models" / "config.json"
LABEL_PATH = current_directory / "models" / "labels.txt"

# 載入模型、詞彙表和標籤映射
model, vocab, label_mapping = load_model(MODEL_PATH, VOCAB_PATH, CONFIG_PATH, LABEL_PATH, device)

# 檢查網址安全性
def check_url_safety(url, text_widget):
    try:
        # 若網址不是以 http:// 或 https:// 開頭，則加上 https://
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        
        # 若網址不是以 https:// 開頭，則印出警告訊息
        if not url.startswith("https://"):
            result = f"【警告】 {url} 使用不安全的協議\n"
            print(f"【警告】 {url} 使用不安全的協議")
            text_widget.insert(tk.END, result)
            return
        
        # 檢查網址中是否含有可疑模式
        suspicious_patterns = ["phishing", "malware", "hack"]
        if any(pattern in url.lower() for pattern in suspicious_patterns):
            result = f"【警告】 {url} 的路徑包含可疑模式\n"
            print(f"【警告】 {url} 的路徑包含可疑模式")
            text_widget.insert(tk.END, result)
            return
        
        # 解析網址並取得主機名稱和 IP 地址
        parsed_url = urlparse(url)
        hostname = parsed_url.hostname
        ip_address = socket.gethostbyname(hostname)
        hostname_from_ip = socket.gethostbyaddr(ip_address)
        
        # 建立 SSL 連線並取得伺服器憑證
        context = ssl.create_default_context()
        context.check_hostname = False
        with context.wrap_socket(socket.socket(), server_hostname=url) as s:
            s.settimeout(5)
            s.connect((hostname, 443))
            cert = s.getpeercert()
        
        # 取得憑證的起始和結束日期
        cert_start_date = cert['notBefore']
        cert_end_date = cert['notAfter']
        
        # 發送 HTTP 請求並檢查回應狀態碼
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            result = f"【安全】 {url} 是安全的\n"
            print(f"【安全】 {url} 是安全的")
            text_widget.insert(tk.END, result)
        else:
            result = f"【警告】 {url} 可能有風險 (狀態碼: {response.status_code}).\n"
            print(f"【警告】 {url} 可能有風險 (狀態碼: {response.status_code}).")
            text_widget.insert(tk.END, result)

    except requests.exceptions.RequestException as e:
        result = f"【錯誤】 {url}: 請求錯誤 ({str(e)})\n"
        print(f"【錯誤】 {url}: 請求錯誤 ({str(e)})")
        text_widget.insert(tk.END, result)
    except ssl.SSLError as ssl_error:
        result = f"【警告】 {url} SSL 握手失敗 ({ssl_error.strerror})\n"
        print(f"【警告】 {url} SSL 握手失敗 ({ssl_error.strerror})")
        text_widget.insert(tk.END, result)
    except socket.timeout:
        result = f"【錯誤】 {url}: 連接超時\n"
        print(f"【錯誤】 {url}: 連接超時")
        text_widget.insert(tk.END, result)
    except socket.error as socket_error:
        result = f"【錯誤】 {url}: 連接錯誤 ({str(socket_error)})\n"
        print(f"【錯誤】 {url}: 連接錯誤 ({str(socket_error)})")
        text_widget.insert(tk.END, result)
    except Exception as e:
        result = f"【錯誤】 {url}: {str(e)}\n"
        print(f"【錯誤】 {url}: {str(e)}")
        text_widget.insert(tk.END, result)

# 預測簡訊類別並顯示結果
def predict_SMS(text, text_widget):
    input_vector = text_to_vector(text)
    input_vector = torch.tensor(input_vector, dtype=torch.float).unsqueeze(0).to(device)
    output = model(input_vector)
    predicted_class = torch.argmax(output).item()
    predicted_probs = output.squeeze().tolist()
    predicted_label = [label for label, index in label_mapping.items() if index == predicted_class][0]
    phone_numbers = re.findall(r'(\(?0\d{1,2}\)?[-\.\s]?\d{3,4}[-\.\s]?\d{3,4})', text)
    urls = re.findall(r'\b(?:https?://)?(?:www\.)?[\w\.-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?(?![\w\.-])\b', text)
    verification_codes = re.findall(r'(?<!\d)(\d{4,6})(?!\d)(?<!/)', text)
    print(f"【簡訊內容】:{text}")
    text_widget.insert(tk.END, f"【簡訊內容】:{text}\n")
    print(f"【預測概率】: {predicted_probs}")
    text_widget.insert(tk.END, f"【預測概率】: {predicted_probs}\n")
    print(f"【預測結果】: {predicted_label}")
    if phone_numbers:
        print(f"【偵測電話】:{phone_numbers}")
    if urls:
        for url in urls:
            threading.Thread(target=check_url_safety, args=(url, text_widget)).start()
            
    if predicted_label == 'Captcha SMS':
        if verification_codes:
            print(f"【驗證碼】:{verification_codes}")
        else:
            print("【驗證碼】:未找到驗證碼。")
    
    return predicted_label, predicted_probs, predicted_class, phone_numbers, urls, verification_codes

# 預測簡訊並顯示結果
def predict_and_display():
    user_input = entry.get()
    if user_input == "":
        messagebox.showinfo("提醒", "請輸入簡訊內容！")
    else:
        predicted_label, predicted_probs, predicted_class, phone_numbers, urls, verification_codes = predict_SMS(user_input, safety_text_box)
        result_label = f"【預測結果】: {predicted_label}\n"
        phone_label = f"【偵測電話】: {phone_numbers}\n" if phone_numbers else ""
        verification_codes_label = f"【驗證碼】: {verification_codes}\n" if predicted_label == 'Captcha SMS' and verification_codes else ""
        result_text = f"{result_label}{phone_label}{verification_codes_label}"
        
        safety_text_box.insert(tk.END, result_text)
        
def open_github():
    webbrowser.open("https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker")
    
def show_version_info():
    messagebox.showinfo("Version Information", f"Current Version: {version}")
    
def save_to_json():
    save_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])

    if save_path:
        with open(save_path, 'w', encoding='utf-8') as file:
            content = safety_text_box.get('1.0', tk.END)
            file.write(content)
            
def open_json_file():
    # 提示用戶選擇要打開的文件
    file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])

    if file_path:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            # 將文件內容顯示在滾動文字框中
            safety_text_box.delete('1.0', tk.END)
            safety_text_box.insert(tk.END, content)

# 切換暗黑模式
def toggle_dark_mode():
    global dark_mode
    dark_mode = not dark_mode
    if dark_mode:
        root.config(bg='#1E1E1E')
        image_label.config(bg='#1E1E1E')
        entry.config(bg='#1F1F1F', fg='white', insertbackground='white')
        button_frame.config(bg='#1E1E1E')
        safety_text_box.config(bg='#1F1F1F', fg='white')
        predict_button.config(bg='#1E1E1E', fg='white')
        clear_button.config(bg='#1E1E1E', fg='white')
        empty_label.config(bg='#1E1E1E')
        menu_bar.config(bg='#1E1E1E')
    else:
        root.config(bg='white')
        image_label.config(bg='white')
        entry.config(bg='white', fg='black')
        button_frame.config(bg='white')
        safety_text_box.config(bg='white', fg='black')
        predict_button.config(bg='white', fg='black')
        clear_button.config(bg='white', fg='black')
        empty_label.config(bg='white')
        menu_bar.config(bg='white') 

dark_mode = False

# 語言選單
def set_language(lang):
    global version
    global predict_button
    global clear_button
    global file_menu
    global language_menu
    global setting_menu
    global root
    global menu_bar
    global help_menu
    
    if lang == "繁體中文":
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
        language_menu.add_command(label="☑ 繁體中文", command=lambda: set_language("繁體中文"))
        language_menu.add_command(label="☐ English", command=lambda: set_language("English"))
        language_menu.add_command(label="☐ 日本語", command=lambda: set_language("日本語"))
        
        help_menu = Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="說明", menu=help_menu)
        help_menu.add_command(label="開啟網站", command=open_github)
        help_menu.add_separator()
        help_menu.add_command(label="版本資訊", command=show_version_info)

        setting_menu = Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="設定", menu=setting_menu)
        setting_menu.add_command(label="開啟暗黑模式", command=toggle_dark_mode)
        
    elif lang == "English":
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
        language_menu.add_command(label="☐ 繁體中文", command=lambda: set_language("繁體中文"))
        language_menu.add_command(label="☑ English", command=lambda: set_language("English"))
        language_menu.add_command(label="☐ 日本語", command=lambda: set_language("日本語"))
        
        help_menu = Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Open Website", command=open_github)
        help_menu.add_separator()
        help_menu.add_command(label="Version Information", command=show_version_info)

        setting_menu = Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Setting", menu=setting_menu)
        setting_menu.add_command(label="Toggle Dark Mode", command=toggle_dark_mode)
    
    elif lang == "日本語":
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
        language_menu.add_command(label="☐ 繁體中文", command=lambda: set_language("繁體中文"))
        language_menu.add_command(label="☐ English", command=lambda: set_language("English"))
        language_menu.add_command(label="☑ 日本語", command=lambda: set_language("日本語"))
        
        help_menu = Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="ヘルプ", menu=help_menu)
        help_menu.add_command(label="ウェブサイトを開く", command=open_github)
        help_menu.add_separator()
        help_menu.add_command(label="バージョン情報", command=show_version_info)

        setting_menu = Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="設定", menu=setting_menu)
        setting_menu.add_command(label="ダークモードを切り替える", command=toggle_dark_mode)
        
# 清除輸入和輸出框內容        
def clear_input():
    entry.delete(0, tk.END)
    safety_text_box.delete('1.0', tk.END)
    
# 建立 tkinter 的 root 視窗
version = "1.0.3"
root = tk.Tk()
root.title(f"Nagato-Sakura-SMS-Checker-GUI-Ver.{version}")
root.geometry("660x540")  
icon_path = current_directory / "assets" / "icon" / f"{version}.ico"
root.iconbitmap(icon_path)

image_path = current_directory / "assets" / "4K" / f"{version}.jpg"
img = Image.open(image_path)

# 計算新的圖片大小以符合視窗比例
img_width, img_height = img.size
window_width = 660
new_img_width = window_width-150
new_img_height = int(img_height * (new_img_width / img_width))

# 重新調整圖片大小
img = img.resize((new_img_width, new_img_height), Image.LANCZOS)
photo = ImageTk.PhotoImage(img)

# 顯示圖片
image_label = tk.Label(root, image=photo)
image_label.pack(fill="both", expand=True, padx=30)

# 輸入框
entry = tk.Entry(root, width=60)
entry.pack(fill="x", expand=True, padx=60)

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

# 建立選單列
menu_bar = Menu(root)
root.config(menu=menu_bar)

set_language("繁體中文")

# GUI 主迴圈
root.mainloop()
