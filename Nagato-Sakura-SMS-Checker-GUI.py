import tkinter as tk
from tkinter import messagebox, scrolledtext
import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from PIL import Image, ImageTk
import re
import requests

# 設備選擇如果有NVIDIA顯卡切換為CUDA
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

    # 初始化模型並將其移到 CUDA 上
    model = SMSClassifier(model_config['input_size'], model_config['hidden_size'], model_config['output_size'])
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    return model, vocab, label_mapping

# 設定檔案路徑
VOCAB_PATH = current_directory /"models"/ "tokenizer.json"
MODEL_PATH = current_directory /"models"/ "SMS_model.bin"
CONFIG_PATH = current_directory /"models"/ "config.json"
LABEL_PATH = current_directory /"models"/ "labels.txt"

# 加载模型、vocab和標籤映射
model, vocab, label_mapping = load_model(MODEL_PATH, VOCAB_PATH, CONFIG_PATH, LABEL_PATH, device)

# 檢查網址安全性的函數
def check_url_safety(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return f"【安全】 {url} 是安全的"
        else:
            return f"【警告】 {url} 可能有風險 (狀態碼: {response.status_code})."
    except Exception as e:
        return f"【錯誤】 {url}: {str(e)}"

# 測試模型
def predict_SMS(text):
    input_vector = text_to_vector(text)
    input_vector = torch.tensor(input_vector, dtype=torch.float).unsqueeze(0).to(device)
    output = model(input_vector)
    predicted_class = torch.argmax(output).item()
    predicted_probs = output.squeeze().tolist()
    predicted_label = [label for label, index in label_mapping.items() if index == predicted_class][0]
    phone_numbers = re.findall(r'(\(?0\d{1,2}\)?[-\.\s]?\d{3,4}[-\.\s]?\d{3,4})', text)
    urls = re.findall(r'\b(?:https?://|www\.)\S+\b', text)
    
    print(f"'{text}'  預測概率: {predicted_probs}")
    print(f"預測結果: {predicted_label}")
    if phone_numbers:
        print(f"偵測電話: {phone_numbers}")
    if urls:
        print(f"偵測網址: {urls}")
        for url in urls:
            check_url_safety(url)
    
    return predicted_label, predicted_probs, predicted_class, phone_numbers, urls

def predict_and_display():
    user_input = entry.get()
    if user_input == "":
        messagebox.showinfo("提醒", "請輸入簡訊內容！")
    else:
        predicted_label, predicted_probs, predicted_class, phone_numbers, urls = predict_SMS(user_input)
        result_label.config(text=f"預測結果: {predicted_label}")
        if phone_numbers:
            phone_label.config(text=f"偵測電話: {phone_numbers}")
        else:
            phone_label.config(text="")  
        if urls:
            url_label.config(text=f"偵測網址: {urls}")
            safety_text = ""
            for url in urls:
                safety_text += check_url_safety(url) + "\n"
            safety_text_box.delete('1.0', tk.END)
            safety_text_box.insert(tk.END, safety_text)
        else:
            url_label.config(text="") 
            safety_text_box.delete('1.0', tk.END)

def toggle_dark_mode():
    if dark_mode_button.config('text')[-1] == '☽':
        root.config(bg='#1E1E1E')
        image_label.config(bg='#1E1E1E')
        entry.config(bg='#1F1F1F', fg='white', insertbackground='white')
        button_frame.config(bg='#1E1E1E')
        result_label.config(bg='#1E1E1E', fg='white')
        phone_label.config(bg='#1E1E1E', fg='white')
        url_label.config(bg='#1E1E1E', fg='white')
        safety_text_box.config(bg='#1F1F1F', fg='white')
        predict_button.config(bg='#1E1E1E', fg='white')
        clear_button.config(bg='#1E1E1E', fg='white')
        dark_mode_button.config(bg='#1E1E1E', fg='white', text='☀')
    else:
        root.config(bg='white')
        image_label.config(bg='white')
        entry.config(bg='white', fg='black')
        button_frame.config(bg='white')
        result_label.config(bg='white', fg='black')
        phone_label.config(bg='white', fg='black')
        url_label.config(bg='white', fg='black')
        safety_text_box.config(bg='white', fg='black')
        predict_button.config(bg='white', fg='black')
        clear_button.config(bg='white', fg='black')
        dark_mode_button.config(bg='white', fg='black', text='☽')
        
def clear_input():
    entry.delete(0, tk.END)
    result_label.config(text="")
    phone_label.config(text="")
    url_label.config(text="")
    safety_text_box.delete('1.0', tk.END)

# 建立主視窗
root = tk.Tk()
root.title("Nagato-Sakura-SMS-Checker-GUI-Ver.1.0.1")
root.geometry("640x480")  

icon_path = current_directory /"assets"/"icon"/"1.0.1.ico"
root.iconbitmap(icon_path)

# 載入圖片
image_path = current_directory /"assets"/"4K"/"1.0.1.png" 
img = Image.open(image_path)
img = img.resize((480, 270), Image.LANCZOS) 
photo = ImageTk.PhotoImage(img)

# 在視窗中顯示圖片
image_label = tk.Label(root, image=photo)
image_label.pack()

# 輸入框
entry = tk.Entry(root, width=60)
entry.pack()

# 預測按鈕和清除按鈕
button_frame = tk.Frame(root)
button_frame.pack()
predict_button = tk.Button(button_frame, text="預測", command=predict_and_display)
predict_button.pack(side="left", padx=5)
clear_button = tk.Button(button_frame, text="清除", command=clear_input)
clear_button.pack(side="left")

# 顯示結果的標籤
result_label = tk.Label(root, text="", font=("Arial", 12))
result_label.pack()

# 顯示電話號碼的標籤
phone_label = tk.Label(root, text="", font=("Arial", 12))
phone_label.pack()

# 顯示網址的標籤
url_label = tk.Label(root, text="", font=("Arial", 12))
url_label.pack()

# 顯示網址安全性的 Text
safety_text_box = scrolledtext.ScrolledText(root, width=60, height=5, wrap=tk.WORD)
safety_text_box.pack()

# Dark Mode 切換按鈕
dark_mode_button = tk.Button(root, text="☽", command=toggle_dark_mode, font=("Arial", 12))
dark_mode_button.place(relx=0.9, rely=0.9, anchor="se")

root.mainloop()
