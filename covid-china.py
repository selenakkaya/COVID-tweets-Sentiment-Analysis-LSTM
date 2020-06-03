import d1original
df = pd.read_csv('COVID-China.csv')
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
        print(count_negative)
        
    elif (np.argmax(sentiment) == 1):
        print("positive")   
        count_positive=count_positive+1
        print(count_positive)
        
   






        
   
print ("positive tweets that was posted on 1st june in China among 1500 random tweets :")
print(count_positive)
print ("negative tweets that was posted on 1st june in China among 1500 random tweets :")
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
plt.title('positive and negative tweets about COVID in China ')  
# Show graphic
plt.show()

