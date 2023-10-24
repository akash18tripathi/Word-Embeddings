# !curl --remote-name-all https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Movies_and_TV.json.gz
# !gzip -d  'reviews_Movies_and_TV.json.gz'
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
import string
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.decomposition import TruncatedSVD
from numpy.linalg import norm
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import pickle

#%matplotlib inline

f = open('reviews_Movies_and_TV.json')

sentences=[]
for i in tqdm(range(40000)):
  data = f.readline()
  t=json.loads(data)
  sentences.append(t['reviewText'])

def preprocess(sentence):
    text = sentence
    text = text.lower()
    text_p = "".join([char for char in text if char not in string.punctuation])
    return text_p

def cosine_dist(u,v):
  return 1-np.dot(u,v)/(norm(u)*norm(v))

def giveTopTenWords(word):
    dist=[]
    for i in df.index:
      dist.append(cosine_dist(df.iloc[we.word2id[word],:],df.iloc[we.word2id[i],:]))

    near = np.argsort(dist)
    ans=[]
    for i in range(11):
      ans.append(we.id2word[near[i]])
    return ans

def display_pca_scatterplot(model, words=None,color='r'):
    word_vectors = np.array([model[w] for w in words])
    pca = PCA(2)
    transformed = pca.fit_transform(word_vectors)
    plt.figure(figsize=(15,15))
    x = transformed[:,0]
    y = transformed[:,1]
    plt.scatter(x,y,c=color)
    for word, (x,y) in zip(words, transformed):
        plt.text(x+0.005, y+0.0009, word)
    plt.savefig("test.png")
    plt.show()


class WordEmbedding:
    def __init__(self, corpus, window_size=5, embedding_dim=100,threshold=5):
        self.corpus = corpus
        self.window_size = window_size
        self.embedding_dim = embedding_dim
        self.vocab = None
        self.co_matrix = None
        self.word2id = None
        self.id2word = None
        self.d={'UNK':0}
        self.threshold=threshold
        
    def build_vocab(self):
        vocab = []
        d={}
        for sentence in self.corpus:
            words = word_tokenize(sentence)
            for word in words:
              vocab .append(word)
              if word in d:
                d[word]+=1
              else:
                d[word]=1
        for k,v in d.items():
          if v>=self.threshold:
            self.d[k]=v
          else:
            
            self.d['UNK']+=1
        self.vocab =list( self.d.keys())
        #replace in corpus
        for j in range(len(self.corpus)):
            sentence=self.corpus[j]
            words = word_tokenize(sentence)
            sent=""
            for i in range(len(words)):
              if words[i] not in self.vocab:
                words[i]='UNK'
            self.corpus[j]=" ".join(words)
        self.word2id = {w:i for i,w in enumerate(self.vocab)}
        self.id2word = {i:w for i,w in enumerate(self.vocab)}
        
    def build_co_matrix(self):
        co_matrix = np.zeros((len(self.vocab), len(self.vocab)))
        for j in tqdm(range(len(self.corpus))):
            sentence = word_tokenize(self.corpus[j])
            for i, word in enumerate(sentence):
                #print(i,word)
                i_start = max(i - self.window_size, 0)
                i_end = min(i + self.window_size, len(sentence) - 1)
                for j in range(i_start, i_end + 1):
                    if i == j:
                        continue
                    co_matrix[self.word2id[word], self.word2id[sentence[j]]] += 1
        self.co_matrix = co_matrix
              
    def build_SVD(self):
      U, s, VT = svds(self.co_matrix,k=self.embedding_dim)
      return U


processed_sentences=[]
count=0
for sentence in sentences:
  sent = sent_tokenize(sentence)
  for s in sent:
    s = preprocess(s)
    processed_sentences.append(s)
    count+=1
    if count==100000:
      break
  if count==100000:
    break
  

we=WordEmbedding(processed_sentences,window_size=5,embedding_dim=100,threshold=5)
we.build_vocab()
print("Vocabulary size:",len(we.vocab))

with open(r"./artifacts/svd_embeddings.pkl", "rb") as input_file:
    embeddings = pickle.load(input_file)


df=pd.DataFrame(embeddings,index=we.vocab)
df.head()

tokens = ['superb','ship','run','films','definitely']
colors=['g','r','b','#FFA500','#FF00FF']
fig, ax = plt.subplots(figsize=(15,15))
for j in range(len(tokens)):
  word=tokens[j]
  d={}
  c=0
  words= giveTopTenWords(word)
  for i in words:
      #print(i)
      d[i]=list(df.loc[i])
      c=c+1
  # display_pca_scatterplot(d,words,colors[j])
  word_vectors = np.array([d[w] for w in words])
  pca = PCA(2)
  transformed = pca.fit_transform(word_vectors)
  x = transformed[:,0]
  y = transformed[:,1]
  ax.scatter(x,y,c=colors[j], label=word)
  for word, (x,y) in zip(words, transformed):
      ax.text(x+0.005, y+0.0009, word)
ax.legend(loc='upper left')
ax.set_title('Multiple Scatter Plots')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
plt.savefig("test.png")
plt.show()

word='titanic'
d={}
c=0
words= giveTopTenWords(word)
for i in words:
    print(i)
    d[i]=list(df.loc[i])
    c=c+1

display_pca_scatterplot(d,words)
