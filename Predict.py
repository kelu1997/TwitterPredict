
import pandas as pd
import numpy as np
import random

import torch

#import torch.utils.data.dataset
import torch.nn as nn
import sys

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
def dataRecoding(url):
    c = pd.read_csv(url)
    X = c
    Y = X['account_type']
    X2 = X.drop(['account_type'], axis=1)
    print(Y.value_counts())
    return X2,Y,X
class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_count = 2
        self.hidden_size = 8
        if self.hidden_count == 2:
            self.fc1 = nn.Linear(in_features=11, out_features=self.hidden_size)
            self.fc2 = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)
            self.fc3 = nn.Linear(in_features=self.hidden_size, out_features=4)
            self.output = nn.Linear(in_features=4, out_features=1)
            self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        if self.hidden_count == 2:
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.dropout(x)
            x = torch.relu(self.fc3(x))
            x = self.output(x)
        return x
def TwitAPI(username):
    # Install tweepy first
    # pip install tweepy
    import tweepy
    import pandas as pd

    # Authenticate keys
    consumer_key = "s1IG611VO8lQdSfySXWRyEmZw"
    consumer_secret = "dN3wBJirRvDuzVTGvbpMyNgf9pYnDM8M3DEvVmzJEraKy0GhtP"
    access_token = "1855909566-QV04OuOXREdsjHhQazLK50vydMzt2qzC4vVC8sy"
    access_secret = "yS9qYUPy3iOWb4EvM21IBIC5CZ8atXGhPO7qndu8VSiAN"

    from tweepy.auth import OAuthHandler
    # OAuth
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)

    # API class
    api = tweepy.API(auth, wait_on_rate_limit=True)

    # Get the User object for twitter...
    # Input screen name
    if username is not None:
        scrn_name = username
    else:
        scrn_name = "Twitter"
    user = api.get_user(screen_name=scrn_name)

    # Models contain the data and some helper methods which we can then use:
    #print(user.screen_name)
    #print(user.followers_count)
    #for friend in user.friends():
    #    print(friend.screen_name)

    # Create a 20d feature vector based on input screen name based on columns of original db
    #print(user.created_at)
    #print(user.default_profile)
    #print(user.default_profile_image)
    #print(user.description)
    #print(user.friends_count)
    #print(user.favourites_count)
    #print(user.followers_count)
    #print(user.geo_enabled)
    #print(user.id)
    #print(user.lang)
    #print(user.location)
    #print(user.profile_background_image_url)
    #print(user.profile_image_url)
    #print(user.screen_name)
    #print(user.statuses_count)
    #print(user.verified)


    # Calculate user account age in days
    from datetime import datetime,timezone

    created_date = user.created_at
    # print(type(created_date)) #class datetime.datetime
    today = datetime.now(timezone.utc)
    delta = today - created_date
    # print(delta.days)

    # ---
    # Calculate average tweets per day
    # Fetching the statuses_count attribute
    # this considers user posts, user replies, and reblogs
    # we will consider this as the "total number of tweets"
    #print("The number of statuses the user has posted are : " + str(user.statuses_count))
    tweets_per_day = user.statuses_count / delta.days
    # print(tweets_per_day)

    data_df = pd.DataFrame(
        {
            "default_profile": [],
            "default_profile_image": [],
            "favourites_count": [],
            "followers_count": [],
            "friends_count": [],
            "statuses_count": [],
            "verified": [],
            "average_tweets_per_day": [],
            "account_age_days": [],
            "screen_name": [],
            "description": [],
        }
    )

   # data_df.loc[0] = [user.created_at, user.default_profile, user.default_profile_image, user.description,
   #                   user.friends_count,
   #                   user.favourites_count, user.followers_count, user.geo_enabled, user.id, user.lang,
   #                   user.location, user.profile_background_image_url, user.profile_image_url, user.screen_name,
   #                   user.statuses_count,
   #                   user.verified, tweets_per_day, delta.days]
    data_df.loc[0] = [user.default_profile, user.default_profile_image,user.favourites_count, user.followers_count,user.friends_count,
                      user.statuses_count, user.verified, tweets_per_day, delta.days,len(user.screen_name),len(user.description)]
    data_df['default_profile'] =  data_df['default_profile'].apply(lambda x: 1 if x == True else 0)
    data_df['default_profile_image'] =  data_df['default_profile_image'].apply(lambda x: 1 if x == True else 0)
    data_df['verified'] =  data_df['verified'].apply(lambda x: 1 if x == True else 0)

    print(data_df)
    return data_df

    import pandas as pd
    import numpy as np
    import requests
    import io


def predict(inputname):
    X = TwitAPI(username=inputname)
    model = ANN()
    url = "Twit.csv"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print(device,torch.version.cuda)
    #model.load_state_dict(torch.load('runs/model_weights.pth',map_location=torch.device(device)))
    model.load_state_dict(torch.load('runs/model_weights.pth'))
    model.to(device)
    model.eval()
    model.double()

    X = X.values
    X_pred = torch.tensor(X,device=device)

    #print(y_pred[:10].data.numpy())
    #print(Y_train[:10])
    y_pred_list1 = []
    y_pred_list2 = []
    with torch.no_grad():
        for i in range(len( X_pred)):
            y_train_pred = model(X_pred[i])
            y_train_pred = torch.sigmoid(y_train_pred)
            y_pred_tag = torch.round(y_train_pred)
            y_pred_list1.append(y_pred_tag.cpu().numpy())
    prediiction = int(y_pred_list1[0])
    if prediiction == 0:
        print("Prediction for " + inputname + " is Bot")
    else:
        print("Prediction for " + inputname + " is Human")
    #traced_model = torch.jit.trace(model, X_pred)
    #traced_model.save('twitTest1.pt')



if __name__ == "__main__":
    if len(sys.argv) > 1:
        predict(sys.argv[1])
    else:
        predict("enemychin")