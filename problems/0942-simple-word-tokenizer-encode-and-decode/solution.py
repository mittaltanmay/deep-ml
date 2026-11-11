import re

def encode(text, vocab):
    clean_text=text.strip()
    regex=r"""([,.:;?_!"()\']|--|\s)"""
    words = [w for w in re.split(regex, clean_text) if w and not w.isspace()]
    encoded=[]
    for word in words:
        encoded.append(vocab[word])
    return encoded 

def decode(ids, vocab):
    inverse_vocab={}
    for key,value in vocab.items():
        inverse_vocab[value]=key
    decoded=" "
    for id in ids:
        decoded+=inverse_vocab[id]
        decoded+=" "
    clean_text=decoded.strip()
    regex=r"""\s+([,.?!"()\'])"""
    ans=re.sub(regex,r"\1",clean_text)
    return ans