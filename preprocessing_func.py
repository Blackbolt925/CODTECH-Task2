import re
import nltk

from nltk.corpus import stopwords
stopwords_list= set(stopwords.words('english'))

TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    return TAG_RE.sub('', text)

class Preprocess():
    def __init__(self):
        pass
    
    def preprocess_text(self, sen):
        sen = sen.lower()
        sentence = remove_tags(sen)# html tags
        sentence = re.sub('[^a-zA-Z]',' ',sentence) #punctuations and numbers
        sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)  # When we remove apostrophe from the word "Mark's", the apostrophe is replaced by an empty space. Hence, we are left with single character "s" that we are removing here.
        sentence = re.sub(r'\s+', ' ', sentence) # Next, we remove all the single characters and replace it by a space which creates multiple spaces in our text. Finally, we remove the multiple spaces from our text as well.
        pattern = re.compile(r'\b(' + r'|'.join(stopwords_list) + r')\b\s*')
        sentence= pattern.sub('',sentence)
        return sentence