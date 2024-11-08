Finetuning_GPTmodel.py
根據不同的"市場情況"去讀取對應的訓練集進行模型微調
請直接修改 market_conditions 這個參數
可以使用不同的個股資料請直接修改 data_name 這個參數


get_data_diff_market.py
可自訂個股名稱取的股市中該股所有資料
取得熊市、牛市...資料；需刪除2024年資料

get_data_history.py
可自訂個股名稱與年份
用於取得個股特定年份資料

model_predict.py
儲存模型預測結果.csv檔案
可視化結果並保存.jpg檔

model_evaluate.py
計算.csv檔評估數值



問題，使用不同的微調模型輸出的結果完全相同
原因，微調訓練只是增加模型對輸出輸入方式的理解無法有效學習預測模式