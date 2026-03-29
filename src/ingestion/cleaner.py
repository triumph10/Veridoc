import re

def clean_text(text):
    if text is None:
        return ""
    text = text.replace("\n"," ")
    text = re.sub(r' +', ' ', text)
    return text.strip()
