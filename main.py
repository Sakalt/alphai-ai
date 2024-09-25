import tkinter as tk
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
import torch

# 1. モデルとトークナイザーのロード（ファインチューニング前）
model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-medium")
tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt2-medium")

# 2. データセットのロード（Hugging Faceの日本語データセットを使用）
dataset = load_dataset("wikipedia", "20220301.jp", split='train[:1%]')

# 3. トークナイズ
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 4. トレーニング設定
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=500,
    evaluation_strategy="epoch"
)

# 5. Trainer の定義
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets
)

# 6. ファインチューニング実行
trainer.train()

# 7. ファインチューニング済みモデルの保存
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

# 8. ファインチューニング後のモデルを再ロード
model = AutoModelForCausalLM.from_pretrained("./fine_tuned_model")
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")

# --- GUI セクション ---

def generate_response(input_text):
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output = model.generate(input_ids, max_length=100, num_return_sequences=1)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# GUI初期設定
def send_message():
    user_input = entry.get()
    if user_input.lower() == 'exit':
        window.quit()
    else:
        bot_response = generate_response(user_input)
        chat_history.config(state=tk.NORMAL)
        chat_history.insert(tk.END, "あなた: " + user_input + "\n")
        chat_history.insert(tk.END, "ボット: " + bot_response + "\n\n")
        chat_history.config(state=tk.DISABLED)
        entry.delete(0, tk.END)

# GUIのウィンドウ作成
window = tk.Tk()
window.title("チャットボット")

# チャット履歴表示
chat_history = tk.Text(window, state=tk.DISABLED)
chat_history.grid(row=0, column=0, columnspan=2)

# ユーザー入力エリア
entry = tk.Entry(window)
entry.grid(row=1, column=0)

# 送信ボタン
send_button = tk.Button(window, text="送信", command=send_message)
send_button.grid(row=1, column=1)

# GUIのループ
window.mainloop()
