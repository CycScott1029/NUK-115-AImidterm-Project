from transformers import GPTNeoForCausalLM, GPT2Tokenizer, Trainer, TrainingArguments
import pandas as pd
import torch
import json

print(f"CUDA version: {torch.version.cuda}")  # 输出CUDA版本
print(f"CuDNN enabled: {torch.backends.cudnn.enabled}")  # 检查CuDNN是否启用

# 参数 {neutral_market、bear_market、bull_market}
market_conditions = "bull_market"

# 检查是否有可用的 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 加载 CSV 数据
file_path = f'./data_set/AAPL_{market_conditions}_data.csv'  # 微调的输入文件
try:
    stock_data = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: File {file_path} not found.")
    exit()

# 选择需要的列
data_columns = ["Open", "High", "Low", "Close", "Volume", "30-day MA", "90-day MA",
                "Volatility", "Daily Returns", "10-day Momentum", "Close (Scaled)", "PE_Ratio"]
data_subset = stock_data[data_columns]

# 定义滑动窗口函数
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

# 创建训练数据集
input_data, target_data = create_sliding_window_data(data_subset)
data = [{"input": input_data[i], "target": target_data[i]} for i in range(len(input_data))]

# 加载 GPT-Neo 模型和分词器，并将模型移到 GPU
model_name = "EleutherAI/gpt-neo-125M"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPTNeoForCausalLM.from_pretrained(model_name).to(device)

# 设置填充符号
tokenizer.pad_token = tokenizer.eos_token

# 自定义数据集
class StockDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = json.dumps(item["input"])
        target_text = json.dumps(item["target"])
        
        # 将输入和标签移动到 device
        inputs = tokenizer(input_text, return_tensors="pt", padding="max_length", max_length=512, truncation=True).to(device)
        labels = tokenizer(target_text, return_tensors="pt", padding="max_length", max_length=512, truncation=True).to(device)
        
        inputs["labels"] = labels["input_ids"].clone()
        return {k: v.squeeze() for k, v in inputs.items()}

# 加载数据集
dataset = StockDataset(data)

# 训练参数，启用混合精度训练 (fp16) 并添加日志记录
training_args = TrainingArguments(
    output_dir=f"./gpt_neo_{market_conditions}_model",
    overwrite_output_dir=True,
    num_train_epochs=50,
    per_device_train_batch_size=2,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=100,  # 频繁记录以监控训练速度
    fp16=True,  # 启用混合精度训练
    dataloader_pin_memory=False
)

# 使用 Trainer 进行微调
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# 开始微调
trainer.train()

# 保存模型
trainer.save_model(f"./gpt_neo_{market_conditions}_model")
tokenizer.save_pretrained(f"./gpt_neo_{market_conditions}_model")

print("模型微调完成并已保存")
