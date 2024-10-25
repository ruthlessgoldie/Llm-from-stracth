# src/utils.py

import re

def load_and_preprocess(file_paths):
    all_text = ""
    for path in file_paths:
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
            # Sadece istenen karakterleri bırak
            text = re.sub(r'[^A-Za-zÇĞİIÖŞÜçğıöşü0-9.,!?;:\-()\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            text = text.lower()  # Türkçe karakterler için lower() kullanıldı
            all_text += text + " "
    return all_text.strip()