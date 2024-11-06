from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import pandas as pd
import torch
import re
import json
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 加載指定市場模型和分詞器的函數
def load_model_and_tokenizer(market_condition):
    model_path = f"./gpt_neo_{market_condition}_model"  # 根據市場條件選擇相應的模型路徑
    model = GPTNeoForCausalLM.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token  # 設置填充符號
    return model, tokenizer

# 定義滑動窗口函數以準備數據
def create_sliding_window_input(data, input_days=30):
    input_sequences = []
    for start in range(len(data) - input_days + 1):
        input_sequence = data.iloc[start:start + input_days].values.flatten().tolist()
        input_sequences.append(input_sequence)
    return input_sequences

# 生成預測
def generate_predictions(model, tokenizer, input_sequences):
    predictions = []
    for input_sequence in input_sequences:
        input_text = json.dumps(input_sequence)
        inputs = tokenizer(input_text, return_tensors="pt", padding="max_length", max_length=512, truncation=True)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 使用正則表達式提取數字，將其轉換為浮點數列表
        predicted_values = re.findall(r"[-+]?\d*\.\d+|\d+", predicted_text)
        predicted_values = [float(val) for val in predicted_values]
        
        # 如果僅需提取第一個值作為預測，取第一個值即可
        if predicted_values:
            predictions.append(predicted_values[0])  # 提取第一個值
        else:
            predictions.append(None)  # 如果無法提取值，設為 None
    return predictions

# 評估模型表現
def evaluate_predictions(true_values, predictions):
    mae = mean_absolute_error(true_values, predictions)
    rmse = np.sqrt(mean_squared_error(true_values, predictions))
    r2 = r2_score(true_values, predictions)
    return mae, rmse, r2

# 主函數
def main():
    # 獲取使用者輸入的市場趨勢
    market_condition = input("Enter market condition (bull_market, bear_market, neutral_market): ").strip()
    
    # 加載相應的模型和分詞器
    try:
        model, tokenizer = load_model_and_tokenizer(market_condition)
    except Exception as e:
        print(f"Error loading model for {market_condition}: {e}")
        return

    # 加載新的 CSV 數據
    file_path = './data_set/AAPL_financial_indicators_2024.csv'
    stock_data = pd.read_csv(file_path)

    # 選擇相同的欄位
    data_columns = ["Open", "High", "Low", "Close", "Volume", "30-day MA", "90-day MA",
                    "Volatility", "Daily Returns", "10-day Momentum", "Close (Scaled)", "PE_Ratio"]
    data_subset = stock_data[data_columns]

    # 準備新數據的輸入
    input_sequences = create_sliding_window_input(data_subset)

    # 生成預測
    predictions = generate_predictions(model, tokenizer, input_sequences)

    # 讀取日期和真實的收盤價格
    dates = pd.to_datetime(stock_data["Date"].iloc[len(stock_data) - len(predictions):].reset_index(drop=True))
    true_values = stock_data["Close"].iloc[len(stock_data) - len(predictions):].reset_index(drop=True)

    # 將結果保存到 CSV
    results_df = pd.DataFrame({
        "Date": dates,
        "True Close Price": true_values,
        "Predicted Close Price": predictions
    })
    output_file = f"{market_condition}_predictions_comparison.csv"
    results_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

    # 評估模型表現
    mae, rmse, r2 = evaluate_predictions(true_values, predictions)
    print(f"MAE: {mae}, RMSE: {rmse}, R2 Score: {r2}")

    # 可視化結果並保存圖片
    plt.figure(figsize=(12, 6))
    plt.plot(dates, true_values, label="True Close Price", color="blue")
    plt.plot(dates, predictions, label="Predicted Close Price", color="red")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.title(f"{market_condition.capitalize()} Market: True vs Predicted Close Price")
    plt.legend()
    
    # 保存圖片
    image_file = f"{market_condition}_predictions_plot.png"
    plt.savefig(image_file)
    print(f"Plot saved to {image_file}")
    
    plt.show()

if __name__ == "__main__":
    main()
