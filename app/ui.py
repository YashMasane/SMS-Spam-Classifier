import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

nltk.download('stopwords')

tfidf = pickle.load(open('model/tfidf_vectorizer.pkl', 'rb'))
model = pickle.load(open('model/mnb.pkl', 'rb'))
porterstemmer = PorterStemmer()

# function for applying preprocessing on text

def preprocessed_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    x = []
    for i in text:
        if i.isalnum():
            x.append(i)

    text = x[:]        
    x.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            x.append(i)

    text = x[:]        
    x.clear()     

    for i in text:  
        x.append(porterstemmer.stem(i))
    
    return " ".join(x)



def make_prediction(sms):
    preprocessed_sms = preprocessed_text(sms)
    input_sms = tfidf.transform([preprocessed_sms])
    return model.predict(input_sms)

