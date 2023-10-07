import torch
from transformers import MarianMTModel, MarianTokenizer

hinglish=[]
def translate_to_hinglish(english_text):
    # Loading the English-to-Hinglish translation model and tokenizer
    model_name = "Helsinki-NLP/opus-mt-en-hi"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    # Tokenizing and translating the English text to Hinglish
    english_text=list(english_text)
    for i in range(len(english_text)):
        input_ids = tokenizer.encode(english_text[i], return_tensors="pt")
        translation = model.generate(input_ids, max_length=100, num_beams=5, early_stopping=True)
        hinglish_translation = tokenizer.decode(translation[0], skip_special_tokens=True)
        hinglish.append(hinglish_translation)
    return hinglish



english_text = ['1.I had about a 30 minute demo just using this new headset.',
                '2.Definitely share your feedback in the comment section.',
                '3.So even if it is a big video, I will clearly mention all the products.',
                '4.I was waiting for my bag.']
hinglish_text = translate_to_hinglish(english_text)

print("Input English Text:")
for text in english_text:
    print(text)

print("\nHinglish Translation:")
for translation in hinglish_text:
    print(translation)
