import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# 讀取股價數據
df = pd.read_csv('./data_set/AAPL_financial_indicators_2024.csv')

# 將數據限制為最近 10 行，轉換為描述格式
data_text = ""
for _, row in df.tail(10).iterrows():
    data_text += f"Date: {row['Date']}, Open: {row['Open']}, High: {row['High']}, Low: {row['Low']}, Close: {row['Close']}, Volume: {row['Volume']}\n"

# 添加生成的預測提示
data_text += "Predict next day: "

# 載入 GPT-2 模型和分詞器
model_name = "gpt2"  # 或使用 "gpt2-medium" 等其他版本
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# 設置填充值
tokenizer.pad_token = tokenizer.eos_token

# 將文本編碼為模型可接受的格式，並限制最大長度以避免超過 GPT-2 模型的限制
inputs = tokenizer(data_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")

# 設置生成參數，調整生成的 tokens 數量
output = model.generate(
    inputs["input_ids"],
    max_new_tokens=30,         # 控制生成的新 tokens 數量
    do_sample=True,            # 啟用隨機抽樣
    temperature=0.7,           # 控制隨機性
    num_return_sequences=1,    # 生成的序列數量
    attention_mask=inputs["attention_mask"],  # 增加 attention mask
    pad_token_id=tokenizer.eos_token_id
)

# 解碼生成的文本
predicted_text = tokenizer.decode(output[0], skip_special_tokens=True)

print("Predicted stock data for future dates:")
print(predicted_text)
