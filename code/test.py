import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

year = 2012

def download_financial_data(ticker_symbol="AAPL", year=year):
    # 定義股票代碼
    ticker = yf.Ticker(ticker_symbol)

    # 獲取股價數據（歷史每日收盤價），擴展查詢範圍
    price_data = ticker.history(period="max")  # 獲取所有可用的歷史數據
    price_data.fillna(method='ffill', inplace=True)  # 向前填充缺失值
    price_data = price_data[price_data['Volume'] > 0]  # 刪除成交量為零的行

    # 計算移動平均線
    price_data['30-day MA'] = price_data['Close'].rolling(window=30).mean()
    price_data['90-day MA'] = price_data['Close'].rolling(window=90).mean()

    # 計算波動率（以日收益的滾動標準差表示）
    price_data['Volatility'] = price_data['Close'].pct_change().rolling(window=30).std()

    # 計算日收益
    price_data['Daily Returns'] = price_data['Close'].pct_change()

    # 創建動量指標（例如，10天回報）
    price_data['10-day Momentum'] = price_data['Close'].pct_change(periods=10)

    # 將 'Close' 價格標準化
    scaler = MinMaxScaler()
    price_data['Close (Scaled)'] = scaler.fit_transform(price_data[['Close']])

    # 刪除由滾動計算產生的 NaN 值
    price_data.dropna(inplace=True)

    # 獲取淨利潤和股份數量
    income_stmt = ticker.financials  # 獲取財務報表
    shares_outstanding = ticker.info['sharesOutstanding']  # 獲取股份數量

    # 初始化 DataFrame 用來儲存 P/E 數據
    pe_ratio = pd.DataFrame(index=price_data.index)  # 使用 price_data 的索引

    # 計算每年的本益比
    net_income = income_stmt.loc['Net Income']  # 獲取淨利潤數據
    years = net_income.index

    # 初始化 P/E Ratio 列
    pe_ratio['PE_Ratio'] = np.nan

    for year_data in years:
        # 計算每股收益（EPS）
        eps = net_income[year_data] / shares_outstanding

        # 獲取當年的價格數據
        year_prices = price_data[price_data.index.year == year_data.year]
        
        # 如果有價格數據，計算 P/E Ratio
        if not year_prices.empty:
            # 使用 .loc 設置 P/E Ratio 值
            pe_ratio.loc[year_prices.index, 'PE_Ratio'] = year_prices['Close'] / eps

    # 合併 price_data 和 pe_ratio
    combined_data = price_data.join(pe_ratio[['PE_Ratio']])

    # 刪除不需要的列（Dividends 和 Stock Splits）
    combined_data.drop(columns=['Dividends', 'Stock Splits'], inplace=True, errors='ignore')

    # 篩選出指定年份的數據
    combined_data_year = combined_data[combined_data.index.year == year]

    # 自動生成文件名，包含年份
    filename = f"{ticker_symbol}_financial_indicators_{year}.csv"
    combined_data_year.to_csv(filename, index=True)

    # print(f"Financial indicators and P/E Ratios for {year} saved to '{filename}'")

    # 顯示指定年份的數據
    # print(combined_data_year)

# 使用函數下載特定年份的數據
download_financial_data(ticker_symbol="AAPL", year=year)
