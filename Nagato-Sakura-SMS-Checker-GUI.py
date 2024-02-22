import tkinter as tk
from tkinter import messagebox, scrolledtext
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
from bs4 import BeautifulSoup
from urllib.parse import urlparse

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

def check_url_safety(url, text_widget):
    try:
        # 若網址不是以 http:// 或 https:// 開頭，則加上 https://
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        
        # 若網址不是以 https:// 開頭，則印出警告訊息
        if not url.startswith("https://"):
            result = f"【警告】 {url} 使用不安全的協議\n"
            text_widget.insert(tk.END, result)
            return
        
        # 檢查網址中是否含有可疑模式
        suspicious_patterns = ["phishing", "malware", "hack"]
        if any(pattern in url.lower() for pattern in suspicious_patterns):
            result = f"【警告】 {url} 的路徑包含可疑模式\n"
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
            text_widget.insert(tk.END, result)
        else:
            result = f"【警告】 {url} 可能有風險 (狀態碼: {response.status_code}).\n"
            text_widget.insert(tk.END, result)

    except requests.exceptions.RequestException as e:
        result = f"【錯誤】 {url}: 請求錯誤 ({str(e)})\n"
        text_widget.insert(tk.END, result)
    except ssl.SSLError as ssl_error:
        result = f"【警告】 {url} SSL 握手失敗 ({ssl_error.strerror})\n"
        text_widget.insert(tk.END, result)
    except socket.timeout:
        result = f"【錯誤】 {url}: 連接超時\n"
        text_widget.insert(tk.END, result)
    except socket.error as socket_error:
        result = f"【錯誤】 {url}: 連接錯誤 ({str(socket_error)})\n"
        text_widget.insert(tk.END, result)
    except Exception as e:
        result = f"【錯誤】 {url}: {str(e)}\n"
        text_widget.insert(tk.END, result)

# 將文字轉換成向量表示
def text_to_vector(text):
    vector = [0] * len(vocab)
    for word in text:
        if word in vocab:
            vector[vocab.index(word)] = 1
    return vector

# 預測簡訊類別並顯示結果
def predict_SMS(text, text_widget):
    input_vector = text_to_vector(text)
    input_vector = torch.tensor(input_vector, dtype=torch.float).unsqueeze(0).to(device)
    output = model(input_vector)
    predicted_class = torch.argmax(output).item()
    predicted_probs = output.squeeze().tolist()
    predicted_label = [label for label, index in label_mapping.items() if index == predicted_class][0]
    phone_numbers = re.findall(r'(\(?0\d{1,2}\)?[-\.\s]?\d{3,4}[-\.\s]?\d{3,4})', text)
    urls = re.findall(r'\b(?:https?://)?(?:www\.)?[\w\.-]+\.[a-zA-Z]{2,}\b', text)
    print(f"'{text}'  預測概率: {predicted_probs}")
    print(f"預測結果: {predicted_label}")
    if phone_numbers:
        print(f"偵測電話: {phone_numbers}")
        
    if urls:
        print(f"偵測網址: {urls}")
        for url in urls:
            threading.Thread(target=check_url_safety, args=(url, text_widget)).start()
    
    return predicted_label, predicted_probs, predicted_class, phone_numbers, urls

# 預測簡訊並顯示結果
def predict_and_display():
    user_input = entry.get()
    if user_input == "":
        messagebox.showinfo("提醒", "請輸入簡訊內容！")
    else:
        predicted_label, predicted_probs, predicted_class, phone_numbers, urls = predict_SMS(user_input, safety_text_box)
        result_label = f"【預測結果】: {predicted_label}"
        phone_label = f"【偵測電話】: {phone_numbers}" if phone_numbers else ""
        result_text = f"{result_label}\n{phone_label}\n" if phone_numbers else f"{result_label}\n"
        
        safety_text_box.insert(tk.END, result_text)

# 切換暗黑模式
def toggle_dark_mode():
    if dark_mode_button.config('text')[-1] == '☽':
        root.config(bg='#1E1E1E')
        image_label.config(bg='#1E1E1E')
        entry.config(bg='#1F1F1F', fg='white', insertbackground='white')
        button_frame.config(bg='#1E1E1E')
        safety_text_box.config(bg='#1F1F1F', fg='white')
        predict_button.config(bg='#1E1E1E', fg='white')
        clear_button.config(bg='#1E1E1E', fg='white')
        dark_mode_button.config(bg='#1E1E1E', fg='white', text='☀')
        empty_label.config(bg='#1E1E1E')
    else:
        root.config(bg='white')
        image_label.config(bg='white')
        entry.config(bg='white', fg='black')
        button_frame.config(bg='white')
        safety_text_box.config(bg='white', fg='black')
        predict_button.config(bg='white', fg='black')
        clear_button.config(bg='white', fg='black')
        dark_mode_button.config(bg='white', fg='black', text='☽')
        empty_label.config(bg='white')

# 清除輸入和輸出框內容        
def clear_input():
    entry.delete(0, tk.END)
    safety_text_box.delete('1.0', tk.END)

# 版本號
version = "1.0.2"

# 建立 tkinter 的 root 視窗
root = tk.Tk()
root.title(f"Nagato-Sakura-SMS-Checker-GUI-Ver.{version}")
root.geometry("640x480")  
icon_path = current_directory / "assets" / "icon" / f"{version}.ico"
root.iconbitmap(icon_path)

image_path = current_directory / "assets" / "4K" / f"{version}.png"
img = Image.open(image_path)
img = img.resize((480, 270), Image.LANCZOS) 
photo = ImageTk.PhotoImage(img)

# 顯示圖片
image_label = tk.Label(root, image=photo)
image_label.pack()

# 輸入框
entry = tk.Entry(root, width=60)
entry.pack()

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
safety_text_box = scrolledtext.ScrolledText(root, width=50, height=6, wrap=tk.WORD, font=("Arial", 11))
safety_text_box.pack()

# 暗黑模式按鈕
dark_mode_button = tk.Button(root, text="☽", command=toggle_dark_mode, font=("Arial", 12))
dark_mode_button.place(relx=0.9, rely=0.9, anchor="se")

# GUI 主迴圈
root.mainloop()
