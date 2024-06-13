import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import json
from tqdm import tqdm
import time

# 檢查是否有可用的 NVIDIA 顯示卡並設置為運算裝置、檢測環境
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("<環境檢測>")
print(f"PyTorch版本 : {torch.__version__}")
print(f"訓練設備 : {device}")

# 定義全域變量
current_directory = os.path.dirname(os.path.abspath(__file__))

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
    
# 生成詞彙表
print("<詞彙表生成>")
json_model_path = os.path.join(current_directory, './data/SMS_data.json')
with open(json_model_path, 'r', encoding='utf-8') as json_file:
    model_data = json.load(json_file)
instructions = [item["instruction"] for item in model_data]
outputs = [item["output"] for item in model_data]
vocab_text = []
for value in model_data:
    if 'weight' in value:
        vocab_text.extend(value)
        
# 確保models資料夾存在，如果不存在就創建它
models_folder = os.path.join(current_directory, 'models')
if not os.path.exists(models_folder):
    os.makedirs(models_folder)
        
# 生成標籤編號將標籤儲存為 labels.txt
print("<類別標籤生成>")
label_mapping = {}
label_count = 0
for label in outputs:
    if label not in label_mapping:
        label_mapping[label] = label_count
        label_count += 1
labels_path = os.path.join(current_directory, './models/labels.txt')
with open(labels_path, 'w') as labels_file:
    for label, index in label_mapping.items():
        labels_file.write(f"{label}: {index}\n")
print(f"標籤已儲存於 {labels_path}")
print(f"類別標籤生成完成 Lables 儲存為 .txt 於 {labels_path}")

# 創建詞彙表 vocab 保存為 tokenizer.json
vocab = list(set(''.join([text for text in instructions])))
tokenizer_path = os.path.join(current_directory, './models/tokenizer.json')
with open(tokenizer_path, 'w') as json_file:
    json.dump(vocab, json_file, indent=2)
print(f"詞彙表生成完成 Tokenizer 儲存為 .json 於 {tokenizer_path}")

# 將文本轉換為向量
def text_to_vector(text):
    vector = [0] * len(vocab)
    for word in text:
        if word in vocab:
            vector[vocab.index(word)] = 1
    return vector

# 將數據轉換為模型可用的格式
train_data = [(text_to_vector(text), label_mapping[label]) for text, label in zip(instructions, outputs)]

print("<訓練開始>")
# 設置模型參數
input_size = len(vocab)
hidden_size = 4096
output_size = len(label_mapping)

# 初始化模型、損失函數和優化器
model = SMSClassifier(input_size, hidden_size, output_size)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)
optimizer = optim.SGD(model.parameters(), lr=5e-2)

# 記錄開始訓練時間
training_start_time = time.time()

# 將數據轉換為 DataLoader 所需的張量
features = [text_to_vector(text) for text in instructions]
labels = [label_mapping[label] for label in outputs]
tensor_x = torch.tensor(features, dtype=torch.float)
tensor_y = torch.tensor(labels, dtype=torch.long)
dataset = TensorDataset(tensor_x, tensor_y)

# 創建 DataLoader 來加載批次數據
batch_size = 32 
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 訓練模型
epochs = 500
for epoch in range(epochs):
    total_loss = 0
    start_time = time.time()

    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    # 輸出每個 epoch 的平均損失
    average_loss = total_loss / len(train_loader)
    # 計算運行時間
    elapsed_time = time.time() - start_time
    # 計算預計完成時間
    eta = (epochs - epoch - 1) * elapsed_time
    # 計算總訓練時間
    total_training_time = time.time() - training_start_time
    
    # 計算空格填充數量
    progress = epoch + 1
    percentage = progress / epochs * 100
    fill_length = int(50 * progress / epochs)
    space_length = 50 - fill_length
    print(f"Processing: {percentage:3.0f}%|{'█' * fill_length}{' ' * space_length}| {progress}/{epochs} [{total_training_time:.2f}<{eta:.2f}, {1 / elapsed_time:.2f}it/s, Loss: {average_loss:.4f}] ")

print("<訓練完成>")

print("<生成模型配置文件>")
model_config = {
    "_name_or_path": "AmanoShizukikun/Nagato-Sakura-SMS-Checker",
    "model_type": "Nagato-Sakura-SMS-Checker",
    "architectures": ["NagatoSakuraSMSCheckerModel"],
    "version": f"NagatoSakuraSMSCheckerModel-{len(train_data)}",
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "learning_rate": optimizer.param_groups[0]['lr'],
}
config_path = os.path.join(current_directory, './models/config.json')
with open(config_path, 'w') as json_file:
    json.dump(model_config, json_file, indent=4)
print(f"模型配置文件生成完成 模型配置文件儲存於 {config_path}")

print("<保存模型>")
model_path = os.path.join(current_directory, './models/SMS_model.bin')
torch.save(model.state_dict(), model_path)
print(f"保存模型完成 模型保存於 {model_path}")

print("<載入測試模式>")
model_path = os.path.join(current_directory, './models/SMS_model.bin')
model = SMSClassifier(input_size, hidden_size, output_size)
model = model.to(device)
model.load_state_dict(torch.load(model_path))

# 測試模型
def predict_SMS(text):
    input_vector = text_to_vector(text)  # 取得文字向量
    input_vector = torch.tensor(input_vector, dtype=torch.float).unsqueeze(0).to(device)  # 調整張量形狀並移到裝置上
    output = model(input_vector)
    predicted_class = torch.argmax(output).item()
    predicted_probs = output.squeeze().tolist()  # 預測機率列表
    predicted_label = [label for label, index in label_mapping.items() if index == predicted_class][0]
    print(f"'{text}'  預測概率: {predicted_probs}")
    print(f"預測結果: {predicted_label}")
    return predicted_label, predicted_probs, predicted_class

# 測試預設範例
test_sentences = [
    "本週6164華興成功獲利40趴 下週強勢飆股已選出 趕緊加賴領取：http://line.me/ti/p/~vi8c", 
    "Hami書城的月讀包「限時下載」一年內會提供超過360本書！會員立即參與投票喜愛的書，有機會抽500元 hamibook.tw/2NQYp", 
    "OPEN POINT會員您好，驗證碼為47385，如非本人操作，建議您立即更改密碼。提醒您！勿將密碼、驗證碼交付他人以防詐騙。", 
    "2023/11/24 14:19 您有0918001824來電1通，提醒您，記得回覆重要電話喔！若已接聽或回電請忽略此則簡訊。"
    ]

for sentence in test_sentences:
    print("<測試開始>")
    predicted_label, predicted_probs, predicted_class = predict_SMS(sentence)
print("<測試完成>")
