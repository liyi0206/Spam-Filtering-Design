Spam Filtering Design
=========
(c) 2014 ly08096

Using 2002 spam email logs from Enron as training data, and 2014 healthcare tweets from Coursolve as testing data, do Spam Filtering using Scikit Learn and NLTK.

Learn from others...
---
The first part is to preprocssing the enron spam datasets. I took the first several rows of codes from here http://nbviewer.ipython.org/github/carljv/Will_it_Python/blob/master/MLFH/CH3/ch3_nltk.ipynb, and the datasets are downloaded here http://spamassassin.apache.org/publiccorpus/ (I got to know it from this discussion http://stackoverflow.com/questions/9876616/which-spam-corpus-i-can-use-in-nltk).

```python
import pandas as pd
import os
import re

data_path=os.path.abspath('./good example/')
spam_path = os.path.join(data_path, 'spam')
spam2_path = os.path.join(data_path, 'spam_2') 
easyham_path = os.path.join(data_path, 'easy_ham')
easyham2_path = os.path.join(data_path, 'easy_ham_2')
hardham_path = os.path.join(data_path, 'hard_ham')
hardham2_path = os.path.join(data_path, 'hard_ham_2')

def get_msgdir(path):
    filelist = os.listdir(path)
    filelist = filter(lambda x: x != 'cmds', filelist)
    all_msgs =[get_msg(os.path.join(path, f)) for f in filelist]
    return all_msgs
def get_msg(path):
    with open(path, 'rU') as con:
        msg = con.readlines()
        first_blank_index = msg.index('\n')
        msg = msg[(first_blank_index + 1): ]
        return ''.join(msg) 

train_spam_messages    = get_msgdir(spam_path)
train_easyham_messages = get_msgdir(easyham_path)
train_easyham_messages = train_easyham_messages[:500]
train_hardham_messages = get_msgdir(hardham_path)

test_spam_messages    = get_msgdir(spam2_path)
test_easyham_messages = get_msgdir(easyham2_path)
test_hardham_messages = get_msgdir(hardham2_path)
```
Preprocessing
-----
This part I preprocess the training and testing data using Scikit Learn. Here I simply used the simple word tokenization. I also tried ngrams - TfidfVectorizer(ngram_range=(1,3)), tokenization by char - TfidfVectorizer(analyzer='char'), and restriction on term frequency - TfidfVectorizer(min_df=3, max_df=5). They are all working very well. 

```python
import string
def cleanit(msgs):
    msgs_new=[]
    for msg in msgs:
        msg = re.sub('3D', '', msg) # Strip out weird '3D' artefacts.
        msg = re.sub('<(.|\n)*?>', ' ', msg)# Strip out html tags and attributes 
        msg = re.sub('&\w+;', ' ', msg) # Strip out html character codes
        msg = re.sub('_+', '_', msg)
        msg=msg.replace('\n','').replace('  ','').strip()
        msg=filter(lambda x: x in string.printable,msg)
        msgs_new.append(msg)
    return msgs_new
train_spam=cleanit(train_spam_messages)
train_ham=cleanit(train_easyham_messages)
train_X=train_spam+train_ham

test_spam=cleanit(test_spam_messages[0:100])
test_ham=cleanit(test_easyham_messages[0:100])
test_X=test_spam+test_ham

vectorizer = TfidfVectorizer() # ngram_range=(1,3)
X_train= vectorizer.fit_transform(train_X) 
X_test = vectorizer.transform(test_X)
y_train=[1 for a in range(500)]+[0 for a in range(500)]
y_test=[1 for a in range(100)]+[0 for a in range(100)]
```

Modeling
--------------
I tried three different models - Logistic Regression, SVM, Stochastic Gradient Boosting. The LR model performs the best, and produce a 99% accuracy. So I use this for new data predicitons. 

```python
from sklearn import linear_model, svm
lr=linear_model.LogisticRegression()
lr.fit(X_train,y_train)
pred=lr.predict(X_test)
accuracy_score(y_test,pred) #0.99 best!

svm=svm.SVC()
svm.fit(X_train,y_train)
pred2=svm.predict(X_test)
accuracy_score(y_test,pred2) #0.955

sgd=linear_model.SGDClassifier()
sgd.fit(X_train,y_train)
pred3=sgd.predict(X_test)
accuracy_score(y_test,pred3) #0.875
```


Predicting
----
Take the June/Blood/Tweets_BleedingDisorders data for example. 

```python
newdata_df = pd.read_csv('./June-2014-08-11/June/Blood/Tweets_BleedingDisorders.csv', header=0)
X_data=newdata_df['content'].values
X_data=cleanit(X_data)
X_new=vectorizer.transform(X_data)
est=lr.predict(X_new)

```


I find the prediciton is not quite accurate by naked eye, this is mainly because the corpus for email spam is very different from tweets spam. There are no free available tweet spam corpus online, so it won't have good application here. But I will follow up this link (http://www.quora.com/Are-there-any-freely-available-twitter-corpora) for updates!


