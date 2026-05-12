# Decision Tree Classifier on Iris Dataset

## Overview
This project implements a Decision Tree Classifier using the Iris Dataset with Scikit-learn.  
The model predicts the species of iris flowers based on sepal and petal measurements.

---

## Dataset
The Iris dataset contains:

- 150 samples
- 3 classes:
  - Setosa
  - Versicolor
  - Virginica
- 4 features:
  - Sepal Length
  - Sepal Width
  - Petal Length
  - Petal Width

---

## Technologies Used

- Python
- Scikit-learn
- NumPy
- Pandas
- Matplotlib

---

## Machine Learning Algorithm

### Decision Tree Classifier
A Decision Tree is a supervised learning algorithm used for classification and regression tasks.  
It works by splitting the dataset into smaller subsets based on feature conditions.

---

## Workflow

1. Import libraries
2. Load Iris dataset
3. Split dataset into training and testing data
4. Train the Decision Tree model
5. Predict output
6. Evaluate model accuracy

---

## Code

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()

X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = DecisionTreeClassifier()

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
