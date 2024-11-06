from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import pandas as pd
import torch
import json
import re

# 加載微調後的模型和分詞器
model_path = "./gpt_neo_stock_model"
model = GPTNeoForCausalLM.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token  # 設置填充符號

# 加載新的 CSV 數據
file_path = './data_set/AAPL_financial_indicators_2024.csv'
stock_data = pd.read_csv(file_path)

# 選擇相同的欄位
data_columns = ["Open", "High", "Low", "Close", "Volume", "30-day MA", "90-day MA",
                "Volatility", "Daily Returns", "10-day Momentum", "Close (Scaled)", "PE_Ratio"]
data_subset = stock_data[data_columns]

# 定義滑動窗口函數以準備數據
def create_sliding_window_input(data, input_days=30):
    input_sequences = []
    
    for start in range(len(data) - input_days + 1):
        input_sequence = data.iloc[start:start + input_days].values.flatten().tolist()
        input_sequences.append(input_sequence)
    
    return input_sequences

# 準備新數據的輸入
input_sequences = create_sliding_window_input(data_subset)

# 生成預測
predictions = []
for input_sequence in input_sequences:
    input_text = json.dumps(input_sequence)
    inputs = tokenizer(input_text, return_tensors="pt", padding="max_length", max_length=512, truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False, pad_token_id=tokenizer.eos_token_id)

    predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 使用正則表達式提取數字，將其轉換為浮點數列表
    predicted_values = re.findall(r"[-+]?\d*\.\d+|\d+", predicted_text)
    predicted_values = [float(val) for val in predicted_values]  # 將提取到的數字轉為浮點數列表
    predictions.append(predicted_values)

# 儲存預測結果
with open("stock_predictions_2024.json", "w") as f:
    json.dump(predictions, f)

print("預測結果已生成並保存在 stock_predictions_2024.json 中")
