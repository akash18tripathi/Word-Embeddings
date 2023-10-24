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
import torch
import torch.nn as nn
import torch.optim as optim
import pickle


def cosine_dist(u,v):
  return 1-np.dot(u,v)/(norm(u)*norm(v))

def giveTopTenWords(word):
    dist=[]
    for i in df.index:
      dist.append(cosine_dist(df.iloc[hp.word2id[word],:],df.iloc[hp.word2id[i],:]))

    near = np.argsort(dist)
    ans=[]
    for i in range(11):
      ans.append(hp.id2word[near[i]])
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

class Helper:
    def __init__(self, corpus, window_size=5, embedding_dim=100,threshold=5):
        self.corpus = corpus
        self.window_size = window_size
        self.embedding_dim = embedding_dim
        self.vocab = None
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
        self.vocab = list(self.d.keys())
        #replace in corpus
        for j in tqdm(range(len(self.corpus))):
            sentence=self.corpus[j]
            words = word_tokenize(sentence)
            sent=""
            for i in range(len(words)):
              if words[i] not in self.vocab:
                words[i]='UNK'
            self.corpus[j]=" ".join(words)
        self.vocab.append('PAD')
        self.word2id = {w:i for i,w in enumerate(self.vocab)}
        self.id2word = {i:w for i,w in enumerate(self.vocab)}
        
    def processInputOutput(self):
        # self.word2id = {w:i for i,w in enumerate(self.vocab)}
        # self.id2word = {i:w for i,w in enumerate(self.vocab)}
        input=[]
        output=[]
        for j in tqdm(range(len(self.corpus))):
            sentence = word_tokenize(self.corpus[j])
            
            for i, word in enumerate(sentence):
                #print(i,word)
                ip=[]
                op=[]
                i_start = i - self.window_size
                i_end = i + self.window_size

                for j in range(i_start, i_end + 1):
                    if i == j:
                        op.append(self.word2id[word])
                        continue
                    if j<0 or j> len(sentence)-1:
                        ip.append(self.word2id['PAD'])
                        continue
                    ip.append(self.word2id[sentence[j]])
                input.append(ip)
                output.append(op)
        return input,output



with open(r"./artifacts/model.pkl", "rb") as input_file:
    model = pickle.load(input_file)

with open(r"./artifacts/embeddings.pkl", "rb") as input_file:
    embeddings = pickle.load(input_file)

with open(r"./artifacts/hp.pkl", "rb") as input_file:
    hp = pickle.load(input_file)


df=pd.DataFrame(embeddings.to('cpu'),index=hp.vocab)
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
