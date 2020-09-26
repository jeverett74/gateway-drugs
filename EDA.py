# Import packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


# import data
df = pd.read_excel("balanced.xlsx")


# print head
df.head()


# describe data
describe = df.describe()

df.shape


# Pearson correlation
corr = df.corr()

# bar charts
fig, ax = plt.subplots()
sns.countplot(df.HARDFLAG,palette='hls')
plt.show()

count_no_sub = len(df[df['MRJFLAG']==0])
count_sub = len(df[df['MRJFLAG']==1])
pct_of_no_sub = count_no_sub/(count_no_sub+count_sub)
print("percentage of no is", pct_of_no_sub*100)
pct_of_sub = count_sub/(count_no_sub+count_sub)
print("percentage of yes is", pct_of_sub*100)

# logistic regression
cols = ['TOBFLAG','ALCFLAG','MRJFLAG']
X=df[cols]
y=df['HARDFLAG']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

# confusion matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()