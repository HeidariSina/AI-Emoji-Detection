!pip install datasets
from datasets import load_dataset
import re, string
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import pandas as pd

# Pre Process Section
def preprocess(text):
    text = text.lower() 
    text=text.strip()  
    text=re.compile('<.*?>').sub('', text) 
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)  
    text = re.sub('\s+', ' ', text)  
    text = re.sub(r'\[[0-9]*\]',' ',text) 
    text=re.sub(r'[^\w\s]', '', str(text).lower().strip())
    text = re.sub(r'\d',' ',text) 
    text = re.sub(r'\s+',' ',text) 
    return text

def stopword(string):
    a= [i for i in string.split() if i not in stopwords.words('english')]
    return ' '.join(a)

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

wl = WordNetLemmatizer()
def lemmatizer(string):
    word_pos_tags = nltk.pos_tag(word_tokenize(string))
    a=[wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) for idx, tag in enumerate(word_pos_tags)]
    return " ".join(a)
    
def finalpreprocess(string):
    return lemmatizer(stopword(preprocess(string)))

def confusion(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred) 
        print(cm)
        plt.matshow(cm)
        plt.title('Confusion matrix')
        plt.colorbar()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

x = []
y = []
dataset = load_dataset('tweet_eval','emoji', split='train')

for textfile in dataset :
  text = finalpreprocess(textfile['text'])
  x.append(text)
  y.append(textfile['label'])

vectorizer = CountVectorizer()
vectors = vectorizer.fit_transform(x)
for i in range (len(y)):
  if y[i] >= 1 :
    y[i] = 1
  else :
    y[i] = 0

X_train, X_test, y_train, y_test = train_test_split(vectors , y, test_size=0.2)
clf_outcome =  MLPClassifier(activation = "relu", max_iter=100 ,hidden_layer_sizes = (2 , 100)).fit(X_train, y_train)
y_outcome_predict = clf_outcome.predict(X_test)
scoreoutcome = clf_outcome.score(X_test, y_test)
print(scoreoutcome)
confusion(y_test , y_outcome_predict)

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import pandas as pd

def confusion(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred) 
        print(cm)
        plt.matshow(cm)
        plt.title('Confusion matrix')
        plt.colorbar()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

X_train, X_test, y_train, y_test = train_test_split(vectors , y, test_size=0.2)
clf_outcome =  MLPClassifier(activation = "relu", max_iter=100 ,hidden_layer_sizes = (2 , 100)).fit(X_train, y_train)
y_outcome_predict = clf_outcome.predict(X_test)
scoreoutcome = clf_outcome.score(X_test, y_test)
print(scoreoutcome)