import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB

data = {
    'cheap' : ['yes','yes','no','yes','no'],
    'promo' : ['yes','no','yes','yes','no'],
    'buy' : ['yes','yes','no','yes','no']
}

df= pd.DataFrame(data)

#step 2 : Encode categorical variables
le_cheap = LabelEncoder()
le_promo = LabelEncoder()
le_buy = LabelEncoder()

df['cheap'] = le_cheap.fit_transform(df['cheap'])
df['promo'] = le_promo.fit_transform(df['promo'])
df['buy'] = le_buy.fit_transform(df['buy'])

X = df[['cheap','promo']]
y = df['buy']

model = CategoricalNB()
model.fit(X,y)
prediction = model.predict(X)

df['Predicted_Buy'] =le_buy.inverse_transform(prediction)
df['Actual_buy'] = le_buy.inverse_transform(y)
print(df)

print(df['promo'])

print(data['promo'])
