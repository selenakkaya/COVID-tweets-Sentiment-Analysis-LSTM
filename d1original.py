
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

data = pd.read_csv('Sentiment.csv')
# Keeping only the neccessary columns
data = data[['text','sentiment']]


data = data[data.sentiment != "Neutral"]
data['text'] = data['text'].apply(lambda x: x.lower())
data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))

print("positive")
print(data[ data['sentiment'] == 'Positive'].size)
print("negative")
print(data[ data['sentiment'] == 'Negative'].size)

for idx,row in data.iterrows():
    row[0] = row[0].replace('rt',' ')

    
max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['text'].values)
X = tokenizer.texts_to_sequences(data['text'].values)
print((data['text'].values))
X = pad_sequences(X)



embed_dim = 128
lstm_out = 196

model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())




#declare the train and test dataset.

Y = pd.get_dummies(data['sentiment']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.33, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

print(X_train)
print(Y_train)
# train the Network.
# number of epochs means how many times you go through your training set. our number of epochs is 7
batch_size = 32
model.fit(X_train, Y_train, epochs = 7, batch_size=batch_size, verbose = 2)

#Extracting a validation set, and measuring score and accuracy.
validation_size = 1500

X_validate = X_test[-validation_size:]
Y_validate = Y_test[-validation_size:]
X_test = X_test[:-validation_size]
Y_test = Y_test[:-validation_size]
score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)
print("score: %.2f" % (score))
print("acc: %.2f" % (acc))

#measuring the number of correct guesses.
pos_cnt, neg_cnt, pos_correct, neg_correct = 0, 0, 0, 0
for x in range(len(X_validate)):
    
    result = model.predict(X_validate[x].reshape(1,X_test.shape[1]),batch_size=1,verbose = 2)[0]
   
    if np.argmax(result) == np.argmax(Y_validate[x]):
        if np.argmax(Y_validate[x]) == 0:
            neg_correct += 1
        else:
            pos_correct += 1
       
    if np.argmax(Y_validate[x]) == 0:
        neg_cnt += 1
    else:
        pos_cnt += 1



print("pos_acc", pos_correct/pos_cnt*100, "%")
print("neg_acc", neg_correct/neg_cnt*100, "%")






df = pd.read_csv('COVID-Turkey.csv')
df=df.dropna(axis=0,how='any')
df['text'] = df['text'].apply(lambda x: x.lower())
df['text'] = df['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))
df['text'] = df['text'].apply(lambda x: x.replace('&amp',' '))
df['text'] = df['text'].apply(lambda x: x.replace('&amp',' '))

#df['sentiment'] = df['sentiment'].set_value)


count_negative=0
count_positive=0
checkpoint=300

for i, row in df.iterrows():
    #print(df['text'])
  
    twt=row['text']
    print(twt)

   # print(twt)
    twt = tokenizer.texts_to_sequences(twt)

    #padding the tweet to have exactly the same shape as `embedding_2` input
    twt = pad_sequences(twt, maxlen=28, dtype='int32', value=0)
    
    print(twt)
    sentiment = model.predict(twt,batch_size=1,verbose = 2)[0]
     

    if(np.argmax(sentiment) == 0):
        print("negative")
        count_negative=count_negative+1
        print(count_positive)
        
    elif (np.argmax(sentiment) == 1):
        print("positive")   
        count_positive=count_positive+1
        print(count_positive)
        
   






        
   
print ("positive tweets that was posted on 1st june in america among 1500 random tweets:")
print(count_positive)
print ("negative tweets that was posted on 1st june in america among 1500 random tweets:")
print(count_negative)

import numpy as np
import matplotlib.pyplot as plt
 
# Make a fake dataset:
height = [count_positive, count_negative]
bars = ('positive', 'negative')
y_pos = np.arange(len(bars))
 
# Create bars
plt.bar(y_pos, height)
 
# Create names on the x-axis
plt.xticks(y_pos, bars)
plt.title('positive and negative tweets about COVID in Turkey ')  
# Show graphic
plt.show()


