import pandas as pd
import json
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

# 加載真實數據和預測數據
true_data_path = './data_set/AAPL_financial_indicators_2024.csv'
true_data = pd.read_csv(true_data_path)

# 加載預測數據
with open("stock_predictions_2024.json", "r") as f:
    predictions = json.load(f)

# 提取真實的收盤價數據
true_close_prices = true_data["Close"].tolist()

# 配對真實數據和預測數據
input_days = 30  # 使用的輸入天數
target_days = 10  # 預測的天數
prediction_dates = true_data["Date"][input_days:len(predictions) + input_days].tolist()

# 檢查並確保長度相同
if len(prediction_dates) > len(predictions):
    prediction_dates = prediction_dates[:len(predictions)]
elif len(predictions) > len(prediction_dates):
    predictions = predictions[:len(prediction_dates)]

# 提取預測的收盤價
predicted_close_prices = [prediction[3] for prediction in predictions]  # 假設 Close 是序列中的第4個值

# 將日期列轉換為日期格式
true_data["Date"] = pd.to_datetime(true_data["Date"])
prediction_dates = pd.to_datetime(prediction_dates)

# 可視化
plt.figure(figsize=(14, 7))

# 畫出歷史收盤價
plt.plot(true_data["Date"], true_close_prices, label="True Close Price", color="blue", alpha=0.7)

# 畫出模型的預測結果
plt.plot(prediction_dates, predicted_close_prices, label="Predicted Close Price", color="red", linestyle="--")

# 設定圖表標題和標籤
plt.title("Historical Stock Prices vs. Predicted Prices")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()

# 設置日期格式僅顯示月份
date_format = DateFormatter("%Y-%m")  # 顯示年份和月份
plt.gca().xaxis.set_major_formatter(date_format)

plt.xticks(rotation=45)
plt.grid(True)

# 儲存圖表為 JPG 文件
plt.savefig("stock_predictions_comparison.jpg", format="jpg", dpi=300)

# 顯示圖表
plt.tight_layout()
plt.show()
