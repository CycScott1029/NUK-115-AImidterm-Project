import pandas as pd
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import Dataset
import torch
import glob

# 讀取並格式化多個CSV數據
def load_and_prepare_data(file_pattern, max_length=128):
    # 讀取所有符合指定模式的CSV文件
    all_files = glob.glob(file_pattern)
    dataframes = []
    
    for file in all_files:
        df = pd.read_csv(file)
        # 格式化文本，包含所有必要的列
        df['text'] = df.apply(lambda row: f"日期: {row['Date']}, 開盤價: {row['Open']}, 最高價: {row['High']}, 最低價: {row['Low']}, 收盤價: {row['Close']}, 成交量: {row['Volume']}, 30日均線: {row['30-day MA']}, 90日均線: {row['90-day MA']}, 波動率: {row['Volatility']}, 日收益: {row['Daily Returns']}, 10日動量: {row['10-day Momentum']}, 本益比: {row['PE_Ratio']}", axis=1)
        dataframes.append(df[['text']])

    # 合併所有DataFrame
    combined_df = pd.concat(dataframes, ignore_index=True)

    # 將數據轉換為Dataset格式
    dataset = Dataset.from_pandas(combined_df)

    # 初始化tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # 設置pad token
    
    # Tokenize數據，並添加labels以便計算loss
    def tokenize_function(examples):
        tokens = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length)
        tokens["labels"] = tokens["input_ids"].copy()  # 將input_ids複製為labels
        return tokens
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset, tokenizer

# 微調GPT-Neo模型
def fine_tune_gpt_neo(file_pattern, output_dir="./finetuned_gpt_neo_timeseries", model_name="EleutherAI/gpt-neo-125M", num_train_epochs=3):
    # 批量載入並格式化數據
    tokenized_dataset, tokenizer = load_and_prepare_data(file_pattern)

    # 加載GPT-Neo模型
    model = GPTNeoForCausalLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))  # 調整詞彙表大小

    # 訓練參數設置
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=2,
        save_steps=500,  # 每500步儲存一次模型
        save_total_limit=2,  # 保留最近的2個模型
        logging_dir='./logs',  # 訓練日誌目錄
        logging_steps=100,
        prediction_loss_only=True,
        fp16=torch.cuda.is_available(),  # 若有GPU則使用混合精度
    )

    # 使用Trainer進行訓練
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    # 開始訓練
    trainer.train()

    # 保存微調後的模型
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"微調後的模型已儲存至：{output_dir}")

# 執行微調
if __name__ == "__main__":
    file_pattern = "./data_set/AAPL_financial_indicators_*.csv"  # 指定文件模式，匹配所有年份的CSV文件
    fine_tune_gpt_neo(file_pattern, output_dir="./finetuned_gpt_neo_timeseries")
