# Import necessary libraries
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Define a function to check if a string represents an integer
def RepresentsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

# Initialize empty lists for target and feature data
Y = []  # Target variable (fraud or not)
X = []  # Feature variables

# Open and read the CSV file
with open("creditcard.csv", "r") as csvhandler:
    data = csv.reader(csvhandler, delimiter=',')
    for row in data:
        # Check if the last column (Class) represents an integer (1 for fraud, 0 for not)
        if RepresentsInt(row[30]):
            # Append feature values (columns 1 to 29) as floats to X
            X.append([float(i) for i in row[1:30]])
            # Append the target value (Class) as a float to Y
            Y.append(float(row[30]))

# Convert lists to NumPy arrays for further processing
X = np.array(X)
Y = np.array(Y)

# Split the data into training and testing sets
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.5, random_state=20)

# Initialize a Logistic Regression classifier
clf = LogisticRegression()

# Train the classifier on the training data
clf.fit(train_x, train_y)

# Print the accuracy score of the classifier on the testing data
print("Score:", clf.score(test_x, test_y))

# Make predictions on the testing data
pred = clf.predict(test_x)

# Print the first prediction and the corresponding true value
print("Predict:", pred[0])
print("Truth:", test_y[0])

# Find the index of the first true fraud in the testing data
num = 0
for i in range(len(test_y)):
    if test_y[i] == 1:
        num = i
        break

# Print the prediction and true value for the first true fraud
print("Predict:", pred[num])
print("Truth:", test_y[num])
