import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# 讀取股價數據
df = pd.read_csv('./data_set/AAPL_financial_indicators_2024.csv')

# 只取收盤價（Close Price）來進行預測
data = df['Close'].values  # 確保您的 CSV 文件中有一個 "Close" 欄位

# 正規化數據以加速模型收斂
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data.reshape(-1, 1))

# 設定輸入步數（n_steps）以根據過去 60 天的數據預測未來一天
n_steps = 60

# 創建訓練數據
X_train, y_train = [], []
for i in range(n_steps, len(scaled_data) - 60):
    X_train.append(scaled_data[i-n_steps:i, 0])
    y_train.append(scaled_data[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

# 調整數據形狀以符合 LSTM 的輸入要求 (samples, time steps, features)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# 建立 LSTM 模型
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

# 編譯模型
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# 訓練模型
model.fit(X_train, y_train, batch_size=64, epochs=50)

# 準備測試數據集
test_data = scaled_data[len(scaled_data) - 60 - n_steps:]
X_test, y_test = [], []
for i in range(n_steps, len(test_data)):
    X_test.append(test_data[i-n_steps:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# 使用模型進行預測
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)  # 將預測值反轉為原始比例

# 可視化結果
plt.figure(figsize=(10, 6))
plt.plot(df['Date'][-len(predictions):], data[-len(predictions):], label='Actual Prices')
plt.plot(df['Date'][-len(predictions):], predictions, label='Predicted Prices')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Stock Price Prediction using LSTM')
plt.legend()
plt.show()
