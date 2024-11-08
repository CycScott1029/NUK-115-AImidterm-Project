import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# 讀取 CSV 檔案
data = pd.read_csv('./result/bear_market_predictions_comparison.csv')

# 取出真實價格和預測價格
true_close = data['True Close Price']
predicted_close = data['Predicted Close Price']

# 計算平均絕對誤差（MAE）
mae = mean_absolute_error(true_close, predicted_close)

# 計算均方根誤差（RMSE）
rmse = np.sqrt(mean_squared_error(true_close, predicted_close))

# 計算 R 平方
r2 = r2_score(true_close, predicted_close)

# 輸出評估結果
print(f"平均絕對誤差 (MAE): {mae}")
print(f"均方根誤差 (RMSE): {rmse}")
print(f"R 平方 (R²): {r2}")
