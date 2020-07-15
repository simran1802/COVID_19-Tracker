import numpy as np
import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt
from plyer import notification
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 

df = pd.read_csv("COVID_19.csv")
ds = df['gender']
u = ds.value_counts()
male = str(u['Male'])
female = str(u['Female'])

std = StandardScaler()
X = std.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=50,criterion='gini',  
random_state=0)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test) 


# def notify_me(title,message):
#     notification.notify(
#         title=title,
#         message=message,
#         app_icon='cave_man_emoticon_emoji_sticker_in_love_icon_123506.ico',  # e.g. 'C:\\icon_32x32.ico'
#         timeout=10,  # seconds
#     )

if __name__ == '__main__':
    # n_title = "Cases of Covid-19 : "
    # l = df['Label']
    # y = l.value_counts().to_list()
    # gndr = ['Number of Male infected :  ' , 'Number of Female infected : ', 'Number of positive cases : ' , 'Number of Negative cases : ']
    # n_message = f"{gndr[0]}  {male}  \n{gndr[1]} {female} \n {gndr[2]} {y[0]} \n {gndr[3]} {y[1]}"
    # notify_me(n_title,  n_message)


    X = df.iloc[:,0:9]
    Y = df.iloc[:,9]

    model = ExtraTreesClassifier()
    model.fit(X,Y)
    print('Accuracy Score :',accuracy_score(y_test, y_pred))

    age=45,
    gender='Male',
    Region1='Solapur',
    Region2='Solapur',
    detected_state='Maharashtra',
    nationality='India',
    Travel_hist='Italy',
    Disease_hist='Null',
    Symptom='Null'

    data=[[age,gender,Region1,Region2,detected_state,nationality,Travel_hist,Disease_hist,Symptom]]
    dfX = pd.DataFrame(data, columns = ['age','gender','Region1 ','Region2','detected_state','nationality','Travel_hist','Disease_hist','Symptom'])
    print(dfX)
    # for c in dfX.columns:
        #print df[c].dtype
        
    # if dfX[c].dtype == "object":
    #     dfX[c] = encodings[c].transform(dfX[c])
    X_test1 = std.transform(dfX)
    y_pred1 = clf.predict(X_test1) 
    ans = encodings['Label'].inverse_transform(y_pred1)
    for dt in ans:
    if dt=='Positive':
        print("Result : High chances of COVID-19")
    else:
        print("Result : You are not suffering from COVID-19")

        