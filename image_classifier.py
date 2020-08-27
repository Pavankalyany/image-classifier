import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
%matplotlib inline

data=pd.read_csv('nearly_mnist.csv')
data.head()

#extract data from the dataset and viewing them upclose
a=data.iloc[3,1:].values

#reshaping the extracted data into a reasonable size
a=a.reshape(28,28).astype('uint8')
plt.imshow(a)

df_x=data.iloc[:,1:]
df_y=data.iloc[:,0]

#creating test and train sizes/batches
x_train,x_test,y_train,y_test=train_test_split(df_x,df_y,test_size=0.2,random_state=4)
x_train.head()
y_train.head()

#call rf classifier
rf=RandomForestClassifier(n_estimators=100)
rf.fit(x_train,y_train)
pred=rf.predict(x_test)
pred

s=y_test.values
count=0
for i in range(len(pred)):
    if pred[i]==s[i]:
        count=count+1
count
len(pred)
#accuracy
(1590/1590)*100
