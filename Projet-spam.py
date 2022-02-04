import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm

pd.set_option('display.max_rows', 2000)
pd.set_option('display.max_columns', 2000)
pd.set_option('display.width', 2000)

#Load du fichier csv contenant les mails
spam = pd.read_csv('spam.csv')

#z et y contiennent les données des colonnes v2 et v1
z = spam['v2']
y = spam["v1"]


z_train, z_test,y_train, y_test = train_test_split(z,y,test_size = 0.2)

cv = CountVectorizer()
features = cv.fit_transform(z_train.values.astype('U'))


model = svm.SVC()
model.fit(features,y_train)

features_test = cv.transform(z_test)
print("Accuracy : {}".format(model.score(features_test,y_test)))

plt.scatter(y_test.index,y_test.values)
plt.show()


