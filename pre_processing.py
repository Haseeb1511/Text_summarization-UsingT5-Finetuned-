import nltk
import string
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download("stopwords")
nltk.download("wordnet")

wn=WordNetLemmatizer()
sw = stopwords.words("english")

def pre_processing(text):
  if isinstance(text,str):
    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+"," ",text)
    text = text.strip().lower()
  else:
    text =""
  return text