from transformers import GPTNeoForCausalLM, GPT2Tokenizer, Trainer, TrainingArguments
import pandas as pd
import torch
import json

# 1. 數據準備

# 加載 CSV 數據
file_path = './data_set/AAPL_financial_indicators_2023.csv'  # 微調的輸入檔案
stock_data = pd.read_csv(file_path)

# 選擇需要的欄位
data_columns = ["Open", "High", "Low", "Close", "Volume", "30-day MA", "90-day MA",
                "Volatility", "Daily Returns", "10-day Momentum", "Close (Scaled)", "PE_Ratio"]
data_subset = stock_data[data_columns]

# 定義滑動窗口函數
def create_sliding_window_data(data, input_days=30, target_days=10):
    sequences = []
    targets = []
    
    total_days = input_days + target_days
    for start in range(len(data) - total_days + 1):
        input_sequence = data.iloc[start:start + input_days].values.flatten().tolist()
        target_sequence = data.iloc[start + input_days:start + total_days].values.flatten().tolist()
        
        sequences.append(input_sequence)
        targets.append(target_sequence)
    
    return sequences, targets

# 創建訓練數據集
input_data, target_data = create_sliding_window_data(data_subset)
data = [{"input": input_data[i], "target": target_data[i]} for i in range(len(input_data))]

# 將數據保存為 JSON 格式（可選）
with open("stock_prediction_data.json", "w") as f:
    json.dump(data, f)

# 加載 GPT-Neo 模型和分詞器
model_name = "EleutherAI/gpt-neo-125M"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPTNeoForCausalLM.from_pretrained(model_name)

# 設置填充符號
tokenizer.pad_token = tokenizer.eos_token  # 將填充符設為 eos_token

# 3. 自定義 Dataset
class StockDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = json.dumps(item["input"])
        target_text = json.dumps(item["target"])
        inputs = tokenizer(input_text, return_tensors="pt", padding="max_length", max_length=512, truncation=True)
        labels = tokenizer(target_text, return_tensors="pt", padding="max_length", max_length=512, truncation=True)
        inputs["labels"] = labels["input_ids"]
        return {k: v.squeeze() for k, v in inputs.items()}

# 加載數據集
dataset = StockDataset(data)

# 4. 訓練參數
training_args = TrainingArguments(
    output_dir="./gpt_neo_stock_model",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs'
)

# 5. 使用 Trainer 進行微調
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# 開始微調
trainer.train()

# 6. 保存模型
trainer.save_model("./gpt_neo_stock_model")
tokenizer.save_pretrained("./gpt_neo_stock_model")

print("模型微調完成並已保存")
