import torch
import torch.nn as nn
import json
from pathlib import Path
import re
import requests
import ssl
import socket
from urllib.parse import urlparse

# 檢查是否有可用的 NVIDIA 顯示卡並設置為運算裝置、檢測環境
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("<環境檢測>")
print(torch.__version__)
print(device)

# 取得目前執行程式的資料夾路徑
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
    # 初始化模型並將其移到 NVIDIA 顯示卡上
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
model, vocab, label_mapping = load_model(MODEL_PATH, VOCAB_PATH, CONFIG_PATH, LABEL_PATH, device)

# 將文本轉換為向量
def text_to_vector(text, vocab):
    vector = [0] * len(vocab)
    for word in text:
        if word in vocab:
            vector[vocab.index(word)] = 1
    return vector

# 預測簡訊類別並顯示結果
def predict_SMS(text):
    input_vector = text_to_vector(text, vocab)
    input_vector = torch.tensor(input_vector, dtype=torch.float).unsqueeze(0).to(device)
    output = model(input_vector)
    predicted_class = torch.argmax(output).item()
    predicted_probs = output.squeeze().tolist()
    predicted_label = [label for label, index in label_mapping.items() if index == predicted_class][0]
    phone_numbers = re.findall(r'(\(?0\d{1,2}\)?[-\.\s]?\d{3,4}[-\.\s]?\d{3,4})', text)
    urls = re.findall(r'\b(?:https?://)?(?:www\.)?[\w\.-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?(?![\w\.-])\b', text)
    verification_codes = re.findall(r'(?<!\d)(\d{4,6})(?!\d)(?<!/)', text)
    result = f"【簡訊內容】:{text}\n"
    result += f"【預測概率】: {predicted_probs}\n"
    result += f"【預測結果】: {predicted_label}\n"
    if phone_numbers:
        result += f"【偵測電話】: {phone_numbers}\n"
    if urls:
        for url in urls:
            result += f"{check_url_safety(url)}\n"
    if predicted_label == 'Captcha SMS':
            if verification_codes:
                result += f"【驗證碼】:{verification_codes}\n"
    return result

# 檢查網址安全性
def check_url_safety(url):
    try:
        # 若網址不是以 http:// 或 https:// 開頭，則加上 https://
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        # 若網址不是以 https:// 開頭，則印出警告訊息
        if not url.startswith("https://"):
            return f"【警告】 {url} 使用不安全的協議"

        # 檢查網址中是否含有可疑模式
        suspicious_patterns = ["phishing", "malware", "hack"]
        if any(pattern in url.lower() for pattern in suspicious_patterns):
            return f"【警告】 {url} 的路徑包含可疑模式"

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
            return f"【安全】 {url} 是安全的"
        else:
            return f"【警告】 {url} 可能有風險 (狀態碼: {response.status_code})."

    except requests.exceptions.RequestException as e:
        return f"【錯誤】 {url}: 請求錯誤 ({str(e)})"
    except ssl.SSLError as ssl_error:
        return f"【警告】 {url} SSL 握手失敗 ({ssl_error.strerror})"
    except socket.timeout:
        return f"【錯誤】 {url}: 連接超時"
    except socket.error as socket_error:
        return f"【錯誤】 {url}: 連接錯誤 ({str(socket_error)})"
    except Exception as e:
        return f"【錯誤】 {url}: {str(e)}"

# 主迴圈 按下 Enter 結束程式或判斷簡訊
print("<測試開始>")
while True:
    user_input = input("請輸入簡訊內容（按下 Enter 結束程式）：")
    if user_input == "":
        break
    else:
        print(predict_SMS(user_input))
