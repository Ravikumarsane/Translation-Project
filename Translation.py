import torch
from transformers import MarianMTModel, MarianTokenizer

def translate_to_hinglish(english_text):
    # Loading the English-to-Hinglish translation model and tokenizer
    model_name = "Helsinki-NLP/opus-mt-en-hi"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    # Tokenizing and translating the English text to Hinglish
    input_ids = tokenizer.encode(english_text, return_tensors="pt")
    translation = model.generate(input_ids, max_length=100, num_beams=5, early_stopping=True)
    hinglish_translation = tokenizer.decode(translation[0], skip_special_tokens=True)

    return hinglish_translation



english_text = "I had about a 30 minute demo just using this new headset."
hinglish_text = translate_to_hinglish(english_text)

print("Input English Text:", english_text)
print("Hinglish Translation:", hinglish_text)
