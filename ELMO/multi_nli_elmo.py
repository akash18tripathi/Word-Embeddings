# Uncomment this!
# !pip install datasets 
from datasets import load_dataset
import re
import string
import nltk
nltk.download('punkt')
import nltk
nltk.download('reuters')
nltk.download('wordnet')
from nltk.corpus import reuters
from nltk import PorterStemmer, WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import keras
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim
import numpy as np
from google.colab import files
from sklearn.metrics import accuracy_score, classification_report

#Unzip below two lines to download glove embeddings
# !wget https://nlp.stanford.edu/data/glove.6B.zip
# !unzip glove.6B.zip -d glove.6B/


stop = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

#Bilstm class
class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_size,embeddings, hidden_size, dropout):
        super(BiLSTM, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding.from_pretrained(embeddings)

        # Bidirectional LSTM layers
        self.lstm1 = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, 
                            num_layers=1, batch_first=True, bidirectional=True)

        # Second LSTM layer
        self.lstm2 = nn.LSTM(input_size=2 * hidden_size, hidden_size=hidden_size, 
                            num_layers=1, batch_first=True, bidirectional=True)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Linear layer
        self.linear = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, X):
        # Embedding layer
        embeddings = self.embedding(X)

        # First LSTM layer
        lstm1_output, _ = self.lstm1(embeddings)
        # Second LSTM layer
        lstm2_output, _ = self.lstm2(lstm1_output)

        # Dropout layer
        lstm2_output = self.dropout(lstm2_output)

        # Linear layer
        output = self.linear(lstm2_output)
        output = torch.transpose(output, 1, 2)
        # output = nn.functional.softmax(output,dim=1)
        return output


class Sentiment_Classifier_MultiNLI(nn.Module):
    def __init__(self, embedding_size):
        super(Sentiment_Classifier_MultiNLI, self).__init__()

        # Embedding layer
        self.s1 = nn.Parameter(torch.ones(1))
        self.s2 = nn.Parameter(torch.ones(1))
        self.s3 = nn.Parameter(torch.ones(1))
        self.alpha = nn.Parameter(torch.ones(1))
        # Linear layer
        self.linear = nn.Linear(embedding_size, 3)

    def forward(self, X):
        # Embedding layer
        embeddings = model.embedding(X.to(device))
        # First LSTM layer
        lstm1_output, _ = model.lstm1(embeddings)
        # Second LSTM layer
        lstm2_output, _ = model.lstm2(lstm1_output)
        # # Dropout layer
        # lstm2_output = self.dropout(lstm2_output)
        output = self.alpha*(self.s1*embeddings + self.s2*lstm1_output + self.s3*lstm2_output)
        #print("layer:",output.shape)
        # Linear layer
        output = self.linear(output)
        output = output.mean(dim=1)
        output = nn.functional.softmax(output)
        return output

def preprocessingText(text, stop=stop):
    text = text.lower() #text to lowercase
    text = re.sub(r'&lt;', '', text) #remove '&lt;' tag
    text = re.sub(r'<.*?>', '', text) #remove html
    text = re.sub(r'[0-9]+', '', text) #remove number
    text = re.sub(r'[^\w\s]', '', text) #remove punctiation
    text = re.sub(r'[^\x00-\x7f]', '', text) #remove non ASCII strings
    for c in ['\r', '\n', '\t'] :
      text = re.sub(c, ' ', text) #replace newline and tab with tabs
    text = re.sub('\s+', ' ', text) #replace multiple spaces with one space
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

def tokenize(sentence):
  tokens = nltk.word_tokenize(sentence)
  return tokens

def getElmoEmbedding(word,embeddings,model1,model2):
    X = model1.embedding(torch.tensor(tokenizer.texts_to_sequences([word])).to(device))
    lstm1_output, _ = model1.lstm1(X)
    lstm2_output, _ = model1.lstm2(lstm1_output)
    output = model2.alpha[0].detach()*(model2.s1[0].detach()*X.detach() + model2.s2[0].detach()*lstm1_output.detach() + model2.s3[0].detach()*lstm2_output.detach())
    return output



#Getting embedding matrix
print("Getting Embedding Matrix...")
embeddings = {}
with open('glove.6B/glove.6B.200d.txt', 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.split()
        word = parts[0]
        vector = np.array(parts[1:], dtype=np.float32)
        embeddings[word] = vector
#Load dataset
data = load_dataset('multi_nli')  

#pick first 1 lakh sentences
print("Processing sentences..")
processed_sentences=[]
y_train=[]
i=0
for d in data['train']:
  txt = preprocessingText(d['premise'])+' <SEP> '+preprocessingText(d['hypothesis'])
  if i==100000:
    break
  i+=1
  y_train.append(d['label'])
  processed_sentences.append(txt)


#Build tokenizer
print("Building tokenizer..")

tokenizer = Tokenizer()
tokenizer.fit_on_texts(processed_sentences)
VOCAB_SIZE = len(tokenizer.word_index) + 1 

#Building vocab and embedding matrix
embedding_matrix = torch.zeros(VOCAB_SIZE, 200)

unk = 0
for i in range(1, VOCAB_SIZE):
  word = tokenizer.index_word[i]
  if word in embeddings.keys():
    embedding_matrix[i] = torch.from_numpy(embeddings[word]).float()
  else:
    unk +=1
print('VOCAB_SIZE : {}'.format(VOCAB_SIZE))
print('TOTAL OF UNKNOWN WORD : {}'.format(unk))



#Preprocessing
input_sequences=[]
for line in processed_sentences:
  token_list = tokenizer.texts_to_sequences([line])[0]    
  input_sequences.append(token_list)

max_sequence_len = max([len(x) for x in input_sequences])
input_sequences_pad = pad_sequences(input_sequences, maxlen=max_sequence_len)

X = input_sequences_pad[:,:-1]
y = input_sequences_pad[:,1:]
X = torch.from_numpy(X)
y = torch.from_numpy(y)




#Filling in Data Loaders
train_dataset = TensorDataset(X, y)
train_dataloader = DataLoader(train_dataset, batch_size=5, shuffle=True)

    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Train Model
# Create an instance of the model
try:
    print("Loding Next word prediction model...")
    model = BiLSTM(VOCAB_SIZE,200, embedding_matrix, 100,0.1)
    model = torch.load('mli_model1.pth',map_location=torch.device('cpu'))
    model=model.to(device)
    print("Loaded model!")
except:
    print("Model not found. Creating a model...This might take some time...")
    model = BiLSTM(VOCAB_SIZE,200, embedding_matrix, 100,0.1).to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # Train the model
    for epoch in range(3):
        total_loss=0  
        for i, (inputs, targets) in enumerate(train_dataloader):
            # Set the model to training mode
            inputs, targets = inputs.to(device), targets.to(device)
            model.train()
            # Zero the gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs)
            #print(outputs.shape, targets.shape)
            outputs = outputs.to(device)
            outputs = outputs.float()
            targets = targets.long()

            # Compute the loss
            loss = criterion(outputs, targets).to(device)
            # Backward pass
            loss.backward()
            total_loss+=loss.item()
            # Update the weights
            optimizer.step()
            # Print the loss every 10 batches
            if i % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {i+1}: Loss {total_loss/(i+1):.4f}")

    #Save model
    torch.save(model, 'mli_model1.pth')
    print("Model saved succesfully!")
    #Download model from colab
    files.download('mli_model1.pth')



  
y_train = keras.utils.to_categorical(y_train, num_classes=3)
y_train = torch.from_numpy(np.asarray(y_train))

train_dataset = TensorDataset(X, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

try:
    print("Loding Sentiment Analysis model prediction model...")
    model2 = Sentiment_Classifier_MultiNLI(200).to(device)
    model2 = torch.load('mli_model2.pth',map_location=torch.device('cpu'))
    print("Loaded!")
except:
    # Create an instance of the model
    # Define the loss function and optimizer
    print("Model not found. Creating a model...This might take some time...")
    model2 = Sentiment_Classifier_MultiNLI(200).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model2.parameters(), lr=0.01)

    # Train the model
    for epoch in range(5):
        total_loss=0  
        for i, (inputs, targets) in enumerate(train_dataloader):
            # Set the model to training mode
            inputs, targets = inputs.to(device), targets.to(device)
            model2.train()
            # Zero the gradients
            optimizer.zero_grad()
            # Forward pass
            # with torch.no_grad():
            #     outputs = model2(inputs).to(device)
            outputs = model2(inputs)
            # outputs = outputs.float()
            targets = targets.view(targets.shape[0],3).float()
            # targets = targets.long()
            #print(outputs.shape, targets.shape)
            # Compute the loss
            loss = criterion(outputs, targets).to(device)
            # Backward pass
            loss.backward()
            total_loss+=loss.item()
            # Update the weights
            optimizer.step()
            # Print the loss every 10 batches
            if i % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {i+1}: Loss {total_loss/(i+1):.4f}")

    torch.save(model2, 'mli_model2.pth')
    print("Model saved succesfully!")
    files.download('mli_model2.pth')



getElmoEmbedding('card',embeddings,model,model2)

print("Processing sentences..")
processed_sentences_test=[]
y_test=[]
for d in data['validation_mismatched']:
  txt = preprocessingText(d['premise'])+' <SEP> '+preprocessingText(d['hypothesis'])

  y_test.append(d['label'])
  processed_sentences_test.append(txt)

input_sequences=[]
for line in processed_sentences_test:
  token_list = tokenizer.texts_to_sequences([line])[0]    
  input_sequences.append(token_list)

max_sequence_len = max([len(x) for x in input_sequences])
input_sequences_pad = pad_sequences(input_sequences, maxlen=max_sequence_len)

X = input_sequences_pad[:,:-1]
X = torch.from_numpy(X)
y_test = torch.tensor(y_test)

test_dataset = TensorDataset(X, y_test)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

y_pred=[]
y_truth=[]
for i, (inputs, targets) in enumerate(test_dataloader):
    # Set the model to training mode
    inputs, targets = inputs.to(device), targets.to(device)
    outputs = model2(inputs).to(device)
    y_pred.append(np.argmax(outputs.cpu().detach()))
    y_truth.append(targets.cpu().detach()[0])

print("Test Accuracy")
print(accuracy_score(y_truth,y_pred))
print("Classification report:")
print(classification_report(y_truth,y_pred))
