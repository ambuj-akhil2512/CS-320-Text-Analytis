# Q.1.	Write a program to read a text file and do the following:			
# a.	Tokenize the text 
# b.	Remove stop words 
# c.	Tag each token with its part of speech
#a. Tokenize the text
import nltk
nltk.download('punkt')
file_content = open("Movie summary dataset.txt").read()
tokens = nltk.word_tokenize(file_content)
print(tokens)

#b. Remove stop words
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')

stop_word=set(stopwords.words('english'))
words=word_tokenize(file_content)
filtered_text = [word for word in words if not word in stop_words]
# after remove stopwords in Movie summary dataset.txt file given by examiner
print(filtered_text)

# Tag each token with its part of speech
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import state_union
from nltk.tokenize  import PunktSentenceTokenizer
nltk.pos_tag(words)

#Q.2 Write a program to read a POS tagged text file and extract all the noun tokens from it.
import nltk
file_content = open("Movie summary dataset.txt").read()#open , read file
tokens = nltk.word_tokenize(file_content) #tokenize
nouns = []
for sentence in tokens:
     for word,pos in nltk.pos_tag(nltk.word_tokenize(str(sentence))):
         if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS'):
             nouns.append(word)
            
#extract all nouns 
print(nouns)
print('Number of nouns in text file :', len(nouns))
#Number of nouns in text file : 315

#Q.3 Use the 20 newsgroups text dataset and classify the news articles using Naïve Bayes Classifier¶
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.datasets import fetch_20newsgroups
data = fetch_20newsgroups()
data.target_names

#classify / defining data 
categories = ['alt.atheism',
 'comp.graphics',
 'comp.os.ms-windows.misc',
 'comp.sys.ibm.pc.hardware',
 'comp.sys.mac.hardware',
 'comp.windows.x',
 'misc.forsale',
 'rec.autos',
 'rec.motorcycles',
 'rec.sport.baseball',
 'rec.sport.hockey',
 'sci.crypt',
 'sci.electronics',
 'sci.med',
 'sci.space',
 'soc.religion.christian',
 'talk.politics.guns',
 'talk.politics.mideast',
 'talk.politics.misc',
 'talk.religion.misc']
#train data on these categories 
train = fetch_20newsgroups(subset='train', categories=categories)
#test data
test = fetch_20newsgroups(subset='test', categories=categories)
# print training data
print(train.data[5]) # test.data give differ
print(len(train.data))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
#creating a model based on multinomial naivebayes
model=make_pipeline(TfidfVectorizer(), MultinomialNB())
#training the model with the train data 
model.fit(train.data, train.target)
#Creating Labels for the test data 
labels=model.predict(test.data)

#Predict Categories on new dat based on trained model
def predict_category(s,train=train,model=model):
    pred=model.predict([s])
    return train.target_names[pred[0]]
  
n=input("Enter the Data  ")
predict_category(n)

# Q.4 . For the attached movie summary dataset do the following:
# a. perform Named Entity Recognition and b. List all the locations present in the text.
#a.	perform Named Entity Recognition and 
import nltk
nltk.download('maxent_ne_chunker')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
# Step Two: Load Data 
sentence = (file_content) 
# Step Three: Tokenise, find parts of speech and chunk words 

for sent in nltk.sent_tokenize(sentence):
  for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
     if hasattr(chunk, 'label'):
        print(chunk.label(), ' '.join(c[0] for c in chunk))
        
        
#b. List all the locations present in the text.
import nltk
file_content = open("Movie summary dataset.txt").read()#open , read file
tokens = nltk.word_tokenize(file_content) #tokenize
for sentence in tokens:
     for word,pos in nltk.pos_tag(nltk.word_tokenize(str(sentence))):
         if (pos == 'GPE'):
             nouns.append(word)
print(nouns)
