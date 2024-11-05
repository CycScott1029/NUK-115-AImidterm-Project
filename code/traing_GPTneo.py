import pandas as pd
from transformers import GPTNeoForCausalLM, AutoTokenizer
import torch
import csv
from datetime import datetime, timedelta

# 讀取整份股價數據
df = pd.read_csv('./data_set/AAPL_financial_indicators_2024.csv')

# 載入 GPT-Neo 模型和分詞器
model_name = "EleutherAI/gpt-neo-1.3B"  # 或者可以使用 "EleutherAI/gpt-neo-2.7B" 模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = GPTNeoForCausalLM.from_pretrained(model_name)

# 設置 pad_token 為 eos_token
tokenizer.pad_token = tokenizer.eos_token

# 初始化 CSV 文件，寫入表頭
output_file = "predicted_report.csv"
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Batch", "Date", "Open", "High", "Low", "Close", "Volume"])

# 設定開始預測的日期
start_date = datetime.strptime("2024-11-05", "%Y-%m-%d")

# 將數據分成多段，每段包含 250 行數據
batch_size = 250
for i in range(0, len(df), batch_size):
    # 取出每段數據並轉換為描述格式
    data_text = ""
    batch_df = df.iloc[i:i+batch_size]
    for _, row in batch_df.iterrows():
        data_text += f"Date: {row['Date']}, Open: {row['Open']}, High: {row['High']}, Low: {row['Low']}, Close: {row['Close']}, Volume: {row['Volume']}\n"

    # 在描述結尾處添加未來多天的空白格式，從指定日期開始遞增
    prediction_days = 10  # 設置要預測的天數
    for j in range(prediction_days):
        next_date = start_date + timedelta(days=j)
        data_text += f"Date: {next_date.strftime('%Y-%m-%d')}, Open: , High: , Low: , Close: , Volume: \n"

    # 將文本編碼為模型可接受的格式
    inputs = tokenizer(data_text, return_tensors="pt", max_length=1024, truncation=True, padding="max_length")

    # 設置生成參數，控制生成的 token 數量
    output = model.generate(
        inputs["input_ids"],
        max_new_tokens=50,       # 調整生成的 tokens 數量以適應多天預測
        do_sample=True,           # 啟用隨機抽樣
        temperature=0.7,          # 控制隨機性
        num_return_sequences=1,   # 生成的序列數量
        attention_mask=inputs["attention_mask"],  # 增加 attention mask
        pad_token_id=tokenizer.eos_token_id
    )

    # 解碼生成的文本
    predicted_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # 解析生成的文本並將結果寫入 CSV
    with open(output_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        for line in predicted_text.splitlines():
            if line.startswith("Date:"):
                parts = line.split(", ")
                if len(parts) < 6:  # 檢查是否包含足夠的字段
                    continue  # 跳過格式不完整的行
                try:
                    date = parts[0].split(": ")[1]
                    open_price = parts[1].split(": ")[1]
                    high = parts[2].split(": ")[1]
                    low = parts[3].split(": ")[1]
                    close = parts[4].split(": ")[1]
                    volume = parts[5].split(": ")[1]
                    writer.writerow([f"Batch {i//batch_size + 1}", date, open_price, high, low, close, volume])
                except IndexError:
                    continue  # 跳過無法解析的行

print("Prediction has been written to predicted_report.csv.")
