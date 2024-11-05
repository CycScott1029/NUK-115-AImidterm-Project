from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import pandas as pd
import torch
import json
from datetime import datetime, timedelta
import re

# 1. 加載微調好的模型和分詞器
model_path = "./gpt_neo_stock_model"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPTNeoForCausalLM.from_pretrained(model_path)

# 添加填充符號
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# 2. 讀取2024年的數據
file_path = './data_set/AAPL_financial_indicators_2024.csv'
stock_data = pd.read_csv(file_path)

# 提取日期欄位的前10個字符
stock_data['Date'] = stock_data['Date'].str[:10]
stock_data['Date'] = pd.to_datetime(stock_data['Date'], errors='coerce')  # 確保日期格式正確

# 選擇需要的列
data_columns = ["Open", "High", "Low", "Close", "Volume", "30-day MA", "90-day MA",
                "Volatility", "Daily Returns", "10-day Momentum", "Close (Scaled)", "PE_Ratio"]
data_subset = stock_data[data_columns]

# 取最新30天數據作為輸入
input_data = data_subset.tail(30).values.flatten().tolist()
input_text = json.dumps(input_data)

# 3. 將輸入數據進行分詞和編碼
inputs = tokenizer(input_text, return_tensors="pt", padding="max_length", max_length=512, truncation=True)
attention_mask = inputs['attention_mask']  # 建立 attention_mask

# 4. 生成預測
model.eval()
with torch.no_grad():
    outputs = model.generate(inputs["input_ids"], max_length=1024, num_return_sequences=1, attention_mask=attention_mask)

# 5. 解碼並處理預測結果
predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# 清理預測文本中的非數字字符，僅保留數字和小數點
predicted_text_cleaned = re.sub(r'[^0-9.\-]', ' ', predicted_text)
predicted_values = [float(x) for x in predicted_text_cleaned.split()]

# 確保預測數據的長度正確
expected_length = 10 * len(data_columns)
if len(predicted_values) < expected_length:
    # 若數據不足，使用最後一個值進行填充
    predicted_values.extend([predicted_values[-1]] * (expected_length - len(predicted_values)))
elif len(predicted_values) > expected_length:
    # 若數據過多，則截取前面的部分
    predicted_values = predicted_values[:expected_length]

# 6. 將預測結果重新格式化為 DataFrame 並保存為 .csv 文件
# 生成日期範圍，從 stock_data 的最後一行日期開始
start_date = pd.to_datetime(stock_data['Date'].dropna().iloc[-1]) + timedelta(days=1)
date_range = [start_date + timedelta(days=i) for i in range(10)]  # 預測 10 天

# 將數據重組為 DataFrame
predicted_df = pd.DataFrame(
    [predicted_values[i:i+len(data_columns)] for i in range(0, len(predicted_values), len(data_columns))],
    columns=data_columns,
    index=date_range
)
predicted_df.index.name = 'Date'  # 設置索引名稱

# 保存為 CSV 文件
output_file_path = './predicted_stock_data.csv'
predicted_df.to_csv(output_file_path)
print(f"預測數據已保存至 {output_file_path}")
