from flask import Flask, request, redirect, url_for, render_template, session, flash, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
# model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

tokenizer = AutoTokenizer.from_pretrained("fine_tuning/fine_tuned_model")
model = AutoModelForCausalLM.from_pretrained("fine_tuning/fine_tuned_model")

app = Flask(__name__)

@app.route("/")
def index():
  return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
  msg = request.form["msg"]
  input = msg
  return get_chat_response(input)

def get_chat_response(text):
  chat_history_ids = torch.tensor([], dtype=torch.long)

  for step in range(5):
    new_user_input_ids = tokenizer.encode(str(text) + tokenizer.eos_token, return_tensors='pt')

    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    return tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

if __name__ == '__main__':
   app.run(debug=True, host='0.0.0.0', port=5000)