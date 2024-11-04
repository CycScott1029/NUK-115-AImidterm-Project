import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 定义股票代码
ticker_symbol = "AAPL"  # 以苹果公司为例
ticker = yf.Ticker(ticker_symbol)

# 获取股价数据（历史每日收盘价），扩展查询范围
price_data = ticker.history(period="max")  # 获取所有可用的历史数据
price_data.fillna(method='ffill', inplace=True)  # 向前填充缺失值
price_data = price_data[price_data['Volume'] > 0]  # 删除成交量为零的行

# 计算移动平均线
price_data['30-day MA'] = price_data['Close'].rolling(window=30).mean()
price_data['90-day MA'] = price_data['Close'].rolling(window=90).mean()

# 计算波动率（以日收益的滚动标准差表示）
price_data['Volatility'] = price_data['Close'].pct_change().rolling(window=30).std()

# 计算日收益
price_data['Daily Returns'] = price_data['Close'].pct_change()

# 创建动量指标（例如，10天回报）
price_data['10-day Momentum'] = price_data['Close'].pct_change(periods=10)

# 将 'Close' 价格标准化
scaler = MinMaxScaler()
price_data['Close (Scaled)'] = scaler.fit_transform(price_data[['Close']])

# 删除由滚动计算产生的 NaN 值
price_data.dropna(inplace=True)

# 获取净利润和股份数量
income_stmt = ticker.financials  # 获取财务报表
shares_outstanding = ticker.info['sharesOutstanding']  # 获取股份数量

# 初始化 DataFrame 用来存储 P/E 数据
pe_ratio = pd.DataFrame(index=price_data.index)  # 使用 price_data 的索引

# 计算每年的本益比
net_income = income_stmt.loc['Net Income']  # 获取净利润数据
years = net_income.index

# 初始化 P/E Ratio 列
pe_ratio['PE_Ratio'] = np.nan

for year in years:
    # 计算每股收益（EPS）
    eps = net_income[year] / shares_outstanding

    # 获取当年的价格数据
    year_prices = price_data[price_data.index.year == year.year]
    
    # 如果有价格数据，计算 P/E Ratio
    if not year_prices.empty:
        # 计算 P/E Ratio
        year_prices['PE_Ratio'] = year_prices['Close'] / eps

        # 将计算的 P/E Ratio 直接填入 pe_ratio DataFrame
        pe_ratio.loc[year_prices.index, 'PE_Ratio'] = year_prices['PE_Ratio']

# 合并 price_data 和 pe_ratio
combined_data = price_data.join(pe_ratio[['PE_Ratio']])

# 删除不需要的列（Dividends 和 Stock Splits）
combined_data.drop(columns=['Dividends', 'Stock Splits'], inplace=True, errors='ignore')

# 筛选出2024年的数据
combined_data_2024 = combined_data[combined_data.index.year == 2024]

# 将结果保存到 CSV 文件
combined_data_2024.to_csv('AAPL_financial_indicators_2024.csv', index=True)  # 保存2024年的数据

print("Financial indicators and P/E Ratios for 2024 saved to 'AAPL_financial_indicators_2024.csv'")

# 显示2024年的数据
print(combined_data_2024)
