import math

import pandas as pd
import numpy as np
import time as Timer
import pickle as pickle
import random

import torch
#import torch.utils.tensorboard
import tensorboard
from ignite.metrics import ClassificationReport
from torch.optim.lr_scheduler import MultiStepLR
#from torch.utils.tensorboard import SummaryWriter
import torch.utils.data.dataset
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
#import datetime as datetime
from datetime import datetime
#TextLogging
import sys


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


#url="https://raw.githubusercontent.com/kelu1997/Datasets/main/twitterhumvbotsUpdated.csv"
url="Twit.csv"
def dataRecoding(url,equalize):
    c = pd.read_csv(url)
    print(c.head())
    load = True
    if load is False:
        #Change Human/Bot to 1/0
        c['account_type'] = c['account_type'].apply(lambda x: 1 if x == "human" else 0)
        # Drop Useless Features/Samples with null Entries
        c2 = c.drop(['Unnamed: 0','created_at','id','profile_background_image_url','profile_image_url'],axis = 1)
        c2 = c2.dropna(subset=['lang','location'])

        X = c2
        X['screen_name_len'] = X['screen_name'].apply(lambda x: len(x)) # Add A Feature off the twitter name length
        X['description_len'] = X['description'].apply(lambda x: len(x)) # Add A Feature off the twitter name length
        X = X.drop(['screen_name','description'],axis = 1) ## Drop Strings

        #LocationCoding
        X['real'] = X['real'].apply(lambda x: 0 if math.isnan(x) else 1)
        X['multiple'] = X['multiple'].apply(lambda x: 0 if math.isnan(x) else 1)
        X['realbutmadeup'] = X['realbutmadeup'].apply(lambda x: 0 if math.isnan(x) else 1)
        X['imaginary'] = X['imaginary'].apply(lambda x: 0 if math.isnan(x) else 1)
        X['BS'] = X['BS'].apply(lambda x: 0 if math.isnan(x) else 1)
        X['missing'] = X['missing'].apply(lambda x: 0 if math.isnan(x) else 1)
        X['otherinfo'] = X['otherinfo'].apply(lambda x: 0 if math.isnan(x) else 1)
        X['uninterpretable'] = X['uninterpretable'].apply(lambda x: 0 if math.isnan(x) else 1)

        X['local_type'] = X['real'].apply(lambda x: 1 if x == 1 else 0)
        X['local_type'] = X['local_type']+(X['multiple']*2)
        X['local_type'] = X['local_type']+(X['realbutmadeup']*3)
        X['local_type'] = X['local_type']+(X['imaginary']*4)
        X['local_type'] = X['local_type']+(X['BS']*5)
        X['local_type'] = X['local_type']+(X['missing']*6)
        X['local_type'] = X['local_type']+(X['otherinfo']*7)
        X['local_type'] = X['local_type']+(X['uninterpretable']*8)
        print(X["local_type"].describe())

        ## Recode True/False entries to 1/0
        X['default_profile'] = X['default_profile'].apply(lambda x: 1 if x == True else 0)
        X['default_profile_image'] = X['default_profile_image'].apply(lambda x: 1 if x == True else 0)
        X['verified'] = X['verified'].apply(lambda x: 1 if x == True else 0)
        X['geo_enabled'] = X['geo_enabled'].apply(lambda x: 1 if x == True else 0)
        X['location'] = X['location'].apply(lambda x: 1 if x != 'unknown' else 0) #Recode to 1 if a "location" provided
        X['lang'] = X['lang'].apply(lambda x: 1 if x == 'en' else 0) ## Change to all english

        X = X.drop(X.loc[X['lang']==0].index)
        if equalize:
            ### Separate the majority and minority classes
            df_minority  = X[X['account_type']==0]
            df_majority = X[X['account_type']==1]
            ### Now, downsamples majority labels equal to the number of samples in the minority class
            df_majority = df_majority.sample(len(df_minority), random_state=0)
            ### concat the majority and minority dataframes
            df = pd.concat([df_majority,df_minority])
            ## Shuffle the dataset to prevent the model from getting biased by similar samples
            X = df.sample(frac=1, random_state=0)

            #X.to_csv("Twit") #To Save
            #### TESTING
            #X = X[X.missing != 1]
            #X = X[X.multiple != 1]
            #X = X.drop(['#NAME?'], axis=1)
            print(X.head())
            Y = X['account_type']
            X2 = X.drop(['account_type'],axis = 1)
            X2 = X2.drop(['local_type'],axis = 1)
            print(Y.value_counts())
            print(X['local_type'].describe())
            print(X['local_type'].value_counts())
    if load is True:
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
        if self.hidden_count == 3:
            self.fc1 = nn.Linear(in_features=14, out_features=self.hidden_size)
            self.fc2 = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)
            self.fc3 = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)
            self.fc4 = nn.Linear(in_features=self.hidden_size, out_features=20)
            self.output = nn.Linear(in_features=20, out_features=1)
            self.dropout = nn.Dropout(0.5)
        if self.hidden_count == 4:
            self.fc1 = nn.Linear(in_features=14, out_features=self.hidden_size)
            self.fc2 = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)
            self.fc3 = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)
            self.fc4 = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)
            self.fc5 = nn.Linear(in_features=self.hidden_size, out_features=20)
            self.output = nn.Linear(in_features=20, out_features=1)
            self.dropout = nn.Dropout(0.15)

    def forward(self, x):
        if self.hidden_count == 2:
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.dropout(x)
            x = torch.relu(self.fc3(x))
            x = self.output(x)
            #x = torch.sigmoid(self.output(x))
        if self.hidden_count == 3:
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.dropout(x)
            x = torch.relu(self.fc3(x))
            x = self.dropout(x)
            x = torch.relu(self.fc4(x))
            #x = torch.sigmoid(self.output(x))
        if self.hidden_count == 4:
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.dropout(x)
            x = torch.relu(self.fc3(x))
            x = self.dropout(x)
            x = torch.relu(self.fc4(x))
            x = self.dropout(x)
            x = torch.relu(self.fc5(x))
            #x = torch.sigmoid(self.output(x))
        return x

def EndEval(model,X,Y,learnRate,time):
    stdoutOrigin = sys.stdout
    sys.stdout = open(str(str("runs"+"/model_weights.pth")), "w")
    model.eval()
    X = X.values
    Y = Y.values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=5)
    X_train = torch.Tensor(X_train)
    X_test = torch.Tensor(X_test)
    Y_train = torch.Tensor(Y_train)
    Y_test = torch.Tensor(Y_test)
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    model.to(device)
    X_train= X_train.to(device)
    Y_train= Y_train.to(device)
    X_test = X_test.to(device)
    Y_test = Y_test.to(device)
    # print(y_pred[:10].data.numpy())
    # print(Y_train[:10])
    y_pred_list1 = []
    y_pred_list2 = []
    with torch.no_grad():
        for i in range(len(X_train)):
            y_train_pred = model(X_train[i])
            y_train_pred = torch.sigmoid(y_train_pred)
            y_pred_tag = torch.round(y_train_pred)
            y_pred_list1.append(y_pred_tag.cpu().numpy())
    y_pred_list1 = [a.squeeze().tolist() for a in y_pred_list1]
    print(confusion_matrix(Y_train, y_pred_list1))
    print(classification_report(Y_train, y_pred_list1, labels=[0, 1]))
    with torch.no_grad():
        for i in range(len(X_test)):
            y_train_pred = model(X_test[i])
            y_train_pred = torch.sigmoid(y_train_pred)
            y_pred_tag = torch.round(y_train_pred)
            y_pred_list2.append(y_pred_tag.cpu().numpy())
    y_pred_list2 = [a.squeeze().tolist() for a in y_pred_list2]
    print(confusion_matrix(Y_test, y_pred_list2))
    print(classification_report(Y_test, y_pred_list2, labels=[0, 1]))
    sys.stdout.close()
    sys.stdout = stdoutOrigin


def Train(model,X,Y,epochs,learnRate,equalized):
    time = datetime.now().strftime('%b%d_%H-%M-%S')
    #writer = SummaryWriter(log_dir=(str("runs/"+str(model.hidden_count)+"X"+str(model.hidden_size)+"LR"+str(learnRate)+"/"+time)))
    X1,Y1=X,Y
    X = X.values
    Y = Y.values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=5)
    X_train = torch.Tensor(X_train)
    X_test = torch.Tensor(X_test)
    Y_train = torch.Tensor(Y_train)
    Y_test = torch.Tensor(Y_test)
    print(X_train.dtype)
    pweight = torch.ones(len(Y_train))
    for i in range(len(Y_train)):
        if Y_train[i]==0:
            pweight[i]=5
    #print(Y_train)
    #print(pweight)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.device_count())
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name(torch.cuda.current_device()))
    print(device)
    print(torch.cuda.is_available())
    #device = "cpu"
    model.to(device)
    pweight = pweight.to(device)
    X_train= X_train.to(device)
    Y_train= Y_train.to(device)
    X_test = X_test.to(device)
    Y_test = Y_test.to(device)
    epoch_acc = 0

    criterion = nn.BCEWithLogitsLoss(pos_weight=pweight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learnRate,weight_decay=1e-2)
    #optimizer = torch.optim.Adam(model.parameters(), lr=learnRate,weight_decay=1e-5)
    mil = 1000000
    #scheduler = MultiStepLR(optimizer, gamma=0.1,milestones=[2*mil,6*mil,10*mil,14*mil,18*mil])
    loss_arr = []

    for i in range(epochs):
        y_hat = model.forward(X_train)
        y_hat = torch.squeeze(y_hat)
        loss = criterion(y_hat, Y_train)
        acc = binary_acc(y_hat, Y_train)
        epoch_acc += acc.item()
        loss_arr.append(loss.item())

        #if i % 100 == 0:
            #writer.add_scalar("Loss/train", loss, i)
            #writer.add_scalar("Acc/train", acc, i)
            #writer.add_scalar("CurrentLr/train", scheduler.get_last_lr()[0], i)


        if i % 10000 == 0:
            print(f'Epoch: {i} Loss: {loss} Acc: {acc} ')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #scheduler.step()

    plt.title('Loss VS Epoch')
    plt.xlabel("Loss")
    plt.xlabel("Epoch")
    plt.plot(loss_arr)
    #plt.show()
    #writer.flush()
    #writer.close()
    torch.save(model.state_dict(), str("runs"+"/model_weights.pth"))
    #EndEval(model,X1,Y1,learnRate,time)


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc

equalize = False
X,Y,df = dataRecoding(url,equalize)

model = ANN()

lr = 0.0001
#Train(model,X,Y,epochs,learnRate=lr)
#####################HYPER#############
epochs = 1000000
for i in range(4,5):
    model = ANN()
    if i == 1:
        lr = 0.01
    if i == 2:
        lr = 0.001
    if i == 3:
        lr = 0.0001
    if i == 4:
        lr = 0.00001
#    print(lr)
    lr = 0.0001
    Train(model, X, Y, epochs, learnRate=lr,equalized=equalize)

Eval = True
if Eval:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print(device,torch.version.cuda)
    #model.load_state_dict(torch.load('runs/model_weights.pth',map_location=torch.device(device)))
    model.load_state_dict(torch.load('runs/model_weights.pth'))
    #model.to(device)
    model.eval()
    model.double()
    X = X.values
    Y = Y.values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=5)
    X_train = torch.tensor(X_train,device=device)
    X_test = torch.tensor(X_test,device=device)
    Y_train = torch.tensor(Y_train,device=device)
    Y_test = torch.tensor(Y_test,device=device)

    y_pred_list1 = []
    y_pred_list2 = []
    with torch.no_grad():
        for i in range(len(X_train)):
            y_train_pred = model(X_train[i])
            y_train_pred = torch.sigmoid(y_train_pred)
            y_pred_tag = torch.round(y_train_pred)
            y_pred_list1.append(y_pred_tag.cpu().numpy())
    y_pred_list1 = [a.squeeze().tolist() for a in y_pred_list1]
    print(confusion_matrix(Y_train.cpu(), y_pred_list1))
    print(classification_report(Y_train.cpu(), y_pred_list1, labels=[0, 1]))
    with torch.no_grad():
        for i in range(len(X_test)):
            y_train_pred = model(X_test[i])
            y_train_pred = torch.sigmoid(y_train_pred)
            y_pred_tag = torch.round(y_train_pred)
            y_pred_list2.append(y_pred_tag.cpu().numpy())
    y_pred_list2 = [a.squeeze().tolist() for a in y_pred_list2]
    print(confusion_matrix(Y_test.cpu(), y_pred_list2))
    print(classification_report(Y_test.cpu(), y_pred_list2, labels=[0, 1]))


