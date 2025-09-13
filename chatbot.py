import queue
import sounddevice as sd
import json
from vosk import Model, KaldiRecognizer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Загружаем модель распознавания речи
model = Model("vosk-model")
rec = KaldiRecognizer(model, 16000)
q = queue.Queue()

# Загружаем лёгкую текстовую модель
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

def callback(indata, frames, time, status):
    if status:
        print(status)
    q.put(bytes(indata))

def listen():
    with sd.RawInputStream(samplerate=16000, blocksize=8000,
                           dtype="int16", channels=1, callback=callback):
        print("🎤 Говори...")
        while True:
            data = q.get()
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                return result.get("text", "")


def chat(text, chat_history_ids=None):
    new_input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors='pt')
    bot_input_ids = new_input_ids if chat_history_ids is None else torch.cat([chat_history_ids, new_input_ids], dim=-1)
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response, chat_history_ids

while True:
    text = listen()
    if text:
        print("👤 Ты сказал:", text)
        answer = chat(text)
        print("🤖 Бот:", answer)
