from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import tweepy
import json
import time
import pandas as pd


#We authenticate ourselves as having a twitter app
#Variables that contains the user credentials to access Twitter API 

consumer_key = '******'
consumer_secret = '***********'
access_token = '*********************'
access_token_secret = '************'

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
 
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)




#We begin searching our query
#Put your search term
searchquery = "covid"

users =tweepy.Cursor(api.search,q=searchquery).items()
count = 0
start = 0
errorCount=0


#here we tell the program how fast to search 
waitquery = 100      #this is the number of searches it will do before resting
waittime = 2.0          # this is the length of time we tell our program to rest
total_number = 15    #this is the total number of queries we want
justincase = 1         #this is the number of minutes to wait just in case twitter throttles us



text = [0] * total_number
secondcount = 0
idvalues = [1] * total_number




while secondcount < total_number:
    try:
        user = next(users)
        count += 1
        
        #We say that after every 100 searches wait 5 seconds
        if (count%waitquery == 0):
            time.sleep(waittime)
            #break

    except tweepy.TweepError:
        #catches TweepError when rate limiting occurs, sleeps, then restarts.
        #nominally 15 minnutes, make a bit longer to avoid attention.
        print ("sleeping....")
        time.sleep(60*justincase)
        user = next(users)
        
        
    except StopIteration:
        break
    try:
        #print "Writing to JSON tweet number:"+str(count)
        text_value = user._json['text']
        language = user._json['lang']
        #print(text_value)
        print(language)
        
        if "RT" not in text_value:
            if language == "en":
                text[secondcount] = text_value
                secondcount = secondcount + 1
                print("current saved is:")
                print(secondcount)

    except UnicodeEncodeError:
        errorCount += 1
        print ("UnicodeEncodeError,errorCount =")+str(errorCount)


print("Creating dataframe:")

d = {"text": text, "id": idvalues}
df = pd.DataFrame(data = d)

df.to_csv('seniorproject.csv', header=True, index=False, encoding='utf-8')

print ("completed")
