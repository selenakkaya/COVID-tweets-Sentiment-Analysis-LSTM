# COVID tweets Sentiment Analysis LSTM
 COVID tweets Sentiment Analysis on LSTM model trained with sentiment140 dataset (https://www.kaggle.com/kazanova/sentiment140)

### for scrapping tweets: tweepy
tweepy (http://docs.tweepy.org/en/latest/)


## For a brief explanation check:
demo-scrapper.ipynb file (tweet scrapping) and 
LSTM-SentimentAnalysis.ipynb file (lstm model)


# Abstract
Social media is used intensively by every segment and people share their real ideas, emotions and opinions through these environments without cencorship. Researches have begun to be conducted over this area with the increaing use of socail media. In particuar, Twitter is one of the most important data sources for human anaylsis and research. The agenda topic and the most talked issues are interpreted especially on Twitter. Owing to this situation, Twitter is used as a source for data in this project. Natural language processing techiniques are used for classifying emotions and extracting the emotion from texts with the technique called ‘Sentiment Analysis’.In this study, sentiment analysis is used to evaluate tweets. The aim is to determine wheter the data is positive or negative. In this study, sentiment anaylsis will be performed using Deep Learning algorithm that is LSTM. The result of algortihms is compared according to use different tweets. In this investigation, corona in ABD, corona in UK, corona in Italy and so on are operated as the topic of tweets. Tweets are downloaded with using Tweepy. Effects of corona on distinct countries are obtained through sentiment analysis on tweets and a comparison between real covid data is compared.

![image](https://user-images.githubusercontent.com/50169967/110125752-f9ebb180-7dc3-11eb-8b49-5e9836757b5d.png)


# Dataset for Trainig

Several different data sets have been tried for this project in order to use train data set. These ways are as an example of using a prepared data set or using a data set generated by ourselves by taking tweets from Twitter and categorizing it like negative or positive.

Data finding process is as follows:

Tweets were getting from Twitter using some keywords to create a positive and negative train set. This way that contemplated for train with these tweets was tried. For the negative data set, tweets that include negative words and characters like “nervouos”, “:(” were got with using Twitter API. Tweets with positive words and characters such as ":)" and "happy" were taken for the positive data set. With this method, it was focused on this, considering that Turkish tweets could be taken and the project could be handled differently. Twitter does not allow hundreds of tweets to be taken at the same time, for example, it was necessary to wait for 1 minute after every 100 tweets. For this reason, creating a large data set took a lot of time and progress was slow. It was decided to experiment with a small number of tweets, but it was not possible to get an optimum result at that point, due to for an example, although most of the tweets with the word "nervous" were negative, there are positive tweets including this word. Sometimes, negative tweet had been perceived as positive. So that, change strategy was to be a esentiality.

Finally, it was decided to use sentiment 140 data from Kaggle for the train operation. This was chosen because it is the most popular data set on the Internet. It was seen that most of the studies performed before this project were analyzed using this data set.

Sentiment 140 data set contains 1,600,000 tweets. This dataset was reached using twitter API. The tweets have been categorized as negative and positive. It is suitable for sentiment analysis. It was prepared using many emotional tweets.





# Getting Tweets with Using Tweepy for Test Data Set

As a first step, it was necessary to create a data set to test the algorithm. Tweets were reached using Tweepy. Tweepy is a Python library for accessing the Twitter API [10]. It ensures to achieve rich and real-time tweet data. OAuth that is an open authorization protocol to authenticate requests is used for Twitter API. To reach the Twitter API, creating and configuring authentication information is obligatory.

First of all, opening a Twitter account and creating a developer account is needed. After that, many questions are asked in the manner as what work will be done with Twitter API or Twitter data. There are separate questions for students and about their projects, and particular questions for those who want to work in different categories. Information is collected about what is intended to be done in detail. Within a period of time, information is checked, and authentication details are given. After that, the application is created and explained in detail in what is desired to be done. API keys and access tokens are accessed. In this way, it is now possible to take tweets.

COVID-19 was chosen as the subject of the tweet due to that analyzing for a current event was aimed. Country-based corona tweets were taken, as people from different countries are more negative or more positive because of the rate of disease increasing or decreasing in their countries, and comparing general negativity and positivity for the countries.

Tweets about a country are wanted to be accessed by typing that country and corona, 1500 tweets with these words are taken. Topics used in this project are;

- Corona and Italy

- Corona and Europe

- Corona and Turkey

- Corona and UK

- Corona and USA

Tweets posted on June 1, 2020 were taken.



