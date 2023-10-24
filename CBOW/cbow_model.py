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

# %matplotlib inline


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
      dist.append(cosine_dist(df.iloc[hp.word2id[word],:],df.iloc[hp.word2id[i],:]))

    near = np.argsort(dist)
    ans=[]
    for i in range(11):
      ans.append(hp.id2word[near[i]])
    return ans

def giveTopTenWordsFromGlove(word):
    dist=[]
    ind={}
    count=0
    for k in embeddings_index.keys():
      dist.append(cosine_dist(embeddings_index[k],embeddings_index[word]))
      ind[count]=k
      count+=1

    near = np.argsort(dist)
    ans=[]
    for i in range(11):
      ans.append(ind[near[i]])
    return ans

def oneHot(vocab_size,x):
  ans=[]
  for k in range(len(x)):
    arr = [0]*vocab_size
    #print(x[k])
    arr[x[k]]=1
    ans.append(arr)
  return torch.FloatTensor(ans)



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

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return torch.tensor(self.X_data[index]), self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)

class MyModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, output_size):
        super(MyModel, self).__init__()
        self.flatten = nn.Flatten()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.linear = nn.Linear(embedding_size, output_size)

    def forward(self, x):
        embedded = self.embedding(x.long())
        embedded = torch.mean(embedded, dim=1)
        output = self.linear(embedded)
        return output   

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


processed_sentences=[]
count=0
for sentence in sentences:
  sent = sent_tokenize(sentence)
  for s in sent:
    s = preprocess(s)
    processed_sentences.append(s)
    count+=1
    if count==40000:
      break
  if count==40000:
    break

print("Processed sentences: ",len(processed_sentences))

hp = Helper(processed_sentences,4)
hp.build_vocab()
print("Vocab Size: ",len(hp.vocab))

input, output = hp.processInputOutput()
input = torch.FloatTensor(input)
data = CustomDataset(input, output)

train_loader = torch.utils.data.DataLoader(data, batch_size=512, shuffle=True)

# Create an instance of the model
model = MyModel(len(hp.vocab),100,len(hp.vocab))

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Training model....")
# Train the model
for epoch in tqdm(range(10)):
    running_loss = 0  
    for ip, op in train_loader:
      optimizer.zero_grad()
      ip = ip.float()
      #op=op.to('cuda')
      pred = model(ip)
      loss = criterion(pred, torch.FloatTensor(oneHot(len(hp.vocab),op[0])))
      loss.backward()
      optimizer.step()
      running_loss += loss.item()
    print("Epoch {} - Training loss: {}".format(epoch, running_loss/len(train_loader)))

print("Training finished")
embeddings = model.embedding.weight.data

df=pd.DataFrame(embeddings,index=hp.vocab)
df.head()

tokens = ['superb','ship','run','films','definitely']
colors=['g','r','b','r','g']
for j in range(len(tokens)):
  word=tokens[j]
  d={}
  c=0
  words= giveTopTenWords(word)
  for i in words:
      print(i)
      d[i]=list(df.loc[i])
      c=c+1

  display_pca_scatterplot(d,words,colors[j])

import pickle
file_name = 'hp.pkl'
with open(file_name, 'wb') as file:
    pickle.dump(hp, file)
    print(f'Object successfully saved to "{file_name}"')

file_name = 'input.pkl'
with open(file_name, 'wb') as file:
    pickle.dump(input, file)
    print(f'Object successfully saved to "{file_name}"')

file_name = 'output.pkl'
with open(file_name, 'wb') as file:
    pickle.dump(output, file)
    print(f'Object successfully saved to "{file_name}"')

file_name = 'embeddings.pkl'
with open(file_name, 'wb') as file:
    pickle.dump(embeddings.to('cpu'), file)
    print(f'Object successfully saved to "{file_name}"')

file_name = 'model.pkl'
with open(file_name, 'wb') as file:
    pickle.dump(model, file)
    print(f'Object successfully saved to "{file_name}"')


# uncomment below
# !wget http://nlp.stanford.edu/data/glove.6B.zip
# !unzip glove*.zip

embeddings_index = {}
f = open('glove.6B.100d.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))
print(giveTopTenWordsFromGlove('titanic'))