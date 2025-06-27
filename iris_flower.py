# Step 1: Import Libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
# Step 2: Load Dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
# Step 3: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Step 4: Create SVM model
model = SVC(kernel='linear')  # You can try 'rbf', 'poly', 'sigmoid'
model.fit(X_train, y_train)
# Step 5: Make predictions
y_pred = model.predict(X_test)
# Step 6: Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))
