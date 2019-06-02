import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def RepresentsInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

Y = []
X = []

with open("creditcard.csv", "r") as csvhandler:
    data = csv.reader(csvhandler, delimiter=',');
    for row in data:
        if (RepresentsInt(row[30])):
            X.append([float(i) for i in row[1:30]]);
            Y.append(float(row[30]));
    
X = np.array(X);
Y = np.array(Y);
    
train_x, test_x, train_y, test_y = train_test_split(X,Y, test_size=0.5,random_state=20);

clf = LogisticRegression();
clf.fit(train_x, train_y);
print("Score:", clf.score(test_x, test_y));

pred =  clf.predict(test_x);
print("Predict:", pred[0] );
print("Truth:", test_y[0]);

# find true fraud for prediction
num = 0;
for i in range(len(test_y)):
    if test_y[i] == 1:
        num = i;
        break;
        
print("Predict:", pred[num] );
print("Truth:", test_y[num]);
