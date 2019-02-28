from xgboost import XGBClassifier
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib

data = pd.read_csv('spam.csv',encoding='latin1')
df = data.drop(columns=data.columns[2:5])
 
x=df['v2']
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(x)
y=df['v1'].map({'ham': 0, 'spam': 1})



model = XGBClassifier(learning_rate=0.3,n_estimators=100,max_depth=8,min_samples_split=2,min_samples_leaf=1,
subsample=1,min_child_weight=3,colsample_bytree=.6,colsample_bylevel=1)
kfold = KFold(n_splits=10,random_state=1)
result = cross_val_score(model,x,y,cv=kfold)

print('Accuracy : %.2f%%'%(result.mean()*100))

joblib.dump(model, 'SMS_spam_model.pkl')