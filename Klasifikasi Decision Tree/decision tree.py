# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt

# Load the modified dataset
dataset = pd.read_csv('C:/Data Mining Kirana/Klasifikasi Decision Tree/Data_Iris.csv', delimiter=';')

# Prepare features and target variable
X = dataset.drop(columns=['Species', 'Id'])  # Features
y = dataset['Species']  # Target

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize Decision Tree Classifier
clf = DecisionTreeClassifier(criterion='entropy')

# Train the model
clf.fit(X_train, y_train)

# Prediction
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calculate Entropy and Information Gain
feature_importances = clf.feature_importances_
print("Feature Importances (Information Gain):", feature_importances)

# Display Decision Tree
plt.figure(figsize=(15, 10))
tree.plot_tree(clf, feature_names=X.columns, class_names=clf.classes_, filled=True)
plt.show()

# Export text representation of the tree
tree_rules = export_text(clf, feature_names=list(X.columns))
print(tree_rules)
