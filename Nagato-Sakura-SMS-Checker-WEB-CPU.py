import torch
import torch.nn as nn
import json
import os
import re
import requests
import ssl
import socket
import gradio as gr
from pathlib import Path
from urllib.parse import urlparse

# 設置運算裝置、檢測環境
device = torch.device("cpu")
current_directory = Path(__file__).resolve().parent
print("<環境檢測>")
print(f"PyTorch 版本: {torch.__version__}")
print(f"運行模式: {device}")

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

# 加載模型和詞彙表
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

# 設定路徑並將模型、詞彙表和標籤映射到裝置
VOCAB_PATH = current_directory / "models" / "tokenizer.json"
MODEL_PATH = current_directory / "models" / "SMS_model.bin"
CONFIG_PATH = current_directory / "models" / "config.json"
LABEL_PATH = current_directory / "models" / "labels.txt"
BLACKLIST_PATH = current_directory / "data" / "blacklist.txt"
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
    result = f"【簡訊內容】: {text}\n【預測概率】: {predicted_probs}\n【預測結果】: {predicted_label}"
    if phone_numbers:
        result += f"\n【偵測電話】: {phone_numbers}"
    if urls:
        for url in urls:
            with open(BLACKLIST_PATH, "r", encoding="utf-8") as file:
                blacklist = set(line.strip() for line in file)
            for blacklisted_url in blacklist:
                if blacklisted_url in urls:
                    result += f"\n【黑名單】: {url} 在黑名單中"
                    break  
            result += "\n" + check_url_safety(url)
    if predicted_label == 'Captcha SMS':
        if verification_codes:
            result += f"\n【驗證碼】: {verification_codes}"
        else:
            result += "\n【驗證碼】: 未找到驗證碼"
    return result

# 檢查網址安全性
def check_url_safety(url):
    try:
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        if not url.startswith("https://"):
            return f"【警告】: {url} 使用不安全的協議"

        suspicious_patterns = ["phishing", "malware", "hack", "top"]
        if any(pattern in url.lower() for pattern in suspicious_patterns):
            return f"【警告】: {url} 的路徑包含可疑模式"

        response = requests.get(url, timeout=3)
        if response.status_code == 200:
            safety_message = f"【安全】: {url} 伺服器已處理請求並正常響應"
        else:
            safety_message = f"【警告】: {url} 可能有風險 (狀態碼: {response.status_code})."
        
    except requests.exceptions.RequestException as e:
        safety_message = f"【錯誤】: {url} 請求錯誤 ({str(e)})"
    except ssl.SSLError as ssl_error:
        safety_message = f"【警告】: {url} SSL 握手失敗 ({ssl_error.strerror})"
    except socket.timeout:
        safety_message = f"【錯誤】: {url} 連接超時"
    except socket.error as socket_error:
        safety_message = f"【錯誤】: {url} 連接錯誤 ({str(socket_error)})"
    except Exception as request_error:
        safety_message = f"【錯誤】: {url} {str(request_error)}"

    try:
        parsed_url = urlparse(url)
        hostname = parsed_url.netloc
        ip_address = socket.gethostbyname(hostname)
        hostname_from_ip = socket.gethostbyaddr(ip_address)
        cert_details = ""

        context = ssl.create_default_context()
        context.check_hostname = False
        with context.wrap_socket(socket.socket(), server_hostname=hostname) as s:
            s.settimeout(5)
            s.connect((hostname, 443))
            cert = s.getpeercert()
        cert_start_date = cert['notBefore']
        cert_end_date = cert['notAfter']
        cert_details = f"【憑證起始日期】: {cert_start_date}\n【憑證結束日期】: {cert_end_date}"
        
    except Exception as e:
        cert_details = f"【錯誤】: {url} {str(e)}"

    return f"{safety_message}\n【主機名稱】: {hostname}\n【IP 位址】: {ip_address}\n【從 IP 位址取得的主機名稱】: {hostname_from_ip}\n{cert_details}"


version = "1.0.6"
image_path = current_directory / "assets" / "4K" / f"{version}.jpg"

examples = [
    "Hami書城的月讀包「限時下載」一年內會提供超過360本書！會員立即參與投票喜愛的書，有機會抽500元 hamibook.tw/2NQYp",
    "OPEN POINT會員您好，驗證碼為47385，如非本人操作，建議您立即更改密碼。提醒您！勿將密碼、驗證碼交付他人以防詐騙",
]

custom_css = """
#logo-img {
    border: none !important;
}
#message {
    font-size: 14px;
    min-height: 300px;
}
"""

# Gradio 界面設定
with gr.Blocks(title=f"Nagato-Sakura-SMS-Checker-GUI-Ver.{version}", analytics_enabled=False, css=custom_css) as demo:
    cid = gr.State("")
    token = gr.State(value=None)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Image(image_path, elem_id="logo-img", show_label=False, show_share_button=False, show_download_button=False)
        with gr.Column(scale=3):
            gr.Markdown(
            """**簡介**: 長門櫻簡訊搜查官(Nagato-Sakura-SMS-Checker) 是「長門櫻計畫」中的一個小分支，是擁有 150 筆簡訊的文本分類模型，具有小巧且輕量化的特性，並且涵蓋4種常見簡訊的不同場景: 詐騙簡訊、廣告簡訊、驗證簡訊、廣告簡訊。
            <br/>
            **注意**: 長門櫻簡訊搜查官模型(Nagato-Sakura-SMS-Checker-Model) 是一個全連接神經網路架構的文本分類模型，並非設計用於聊天使用。  
            <br/>
            **模型**: [Nagato Sakura SMS Checker Model](https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker)
            <br/> 
            **開發者**:
            <a href="https://github.com/AmanoShizukikun/Nagato-Sakura-SMS-Checker/graphs/contributors" target="_blank">
            <img src="https://contrib.rocks/image?repo=AmanoShizukikun/Nagato-Sakura-SMS-Checker" />
            """
            )
            
    with gr.Column():
        with gr.Row():
            chatbot = gr.Chatbot(show_label=False, show_share_button=False, show_copy_button=True)
        
        with gr.Row():
            user_message = gr.Textbox(lines=1, placeholder="請輸入簡訊內容...", label="Input", show_label=False)

        with gr.Row():
            submit_button = gr.Button("預測")
            clear_button = gr.Button("清除")

        history = gr.State([])

        def process_message(user_message, history):
            prediction = predict_SMS(user_message)
            history = history + [(user_message, prediction)]
            return history, history

        def clear_all():
            return [], "", []

        user_message.submit(process_message, inputs=[user_message, history], outputs=[chatbot, history])
        submit_button.click(process_message, inputs=[user_message, history], outputs=[chatbot, history])
        clear_button.click(clear_all, outputs=[chatbot, user_message, history])

        with gr.Row():
            gr.Examples(
                examples=examples,
                inputs=user_message,
                cache_examples=False,
                outputs=[chatbot],
                examples_per_page=100
            )

if __name__ == "__main__":
    try:
        gradio_share = os.environ.get("GRADIO_SHARE", "0").lower() in ["true", "1"]
        server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
        demo.queue(api_open=True, max_size=40).launch(show_api=True, share=gradio_share, server_name=server_name, inbrowser=True)
    except Exception as e:
        print(f"Error: {e}")
