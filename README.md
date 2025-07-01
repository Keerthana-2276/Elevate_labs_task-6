🚀 K-Nearest Neighbors (KNN) Classification on Iris Dataset
This project demonstrates classification using the KNN algorithm on the Iris dataset using Python and scikit-learn in Google Colab.

📁 Dataset
We use the Iris dataset containing sepal and petal measurements for three flower species: Setosa, Versicolor, and Virginica.

🛠️ Tasks Performed

🔍 1. Import Dataset & Normalize Features
- Load the Iris dataset using `sklearn.datasets.load_iris`.
- Use only the first 2 features (Sepal Length and Sepal Width) for visualization.
- Normalize the features using `StandardScaler`.

```python
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris.data[:, :2]  # Only first 2 features
y = iris.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
```

🔢 2. Train and Evaluate KNN for Multiple K
Train KNN classifier for K=1 to 10 and evaluate using accuracy score.

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

best_k = 1
best_acc = 0

for k in range(1, 11):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"K = {k} ➤ Accuracy: {acc:.2f}")
    if acc > best_acc:
        best_acc = acc
        best_k = k
```

📊 3. Evaluate Best Model Using Accuracy and Confusion Matrix

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print(f"\n✅ Best K = {best_k} with Accuracy = {accuracy_score(y_test, y_pred):.2f}")

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot()
```

🧠 4. Visualize Decision Boundaries

```python
import numpy as np
import matplotlib.pyplot as plt

h = 0.02
x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Set1)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, edgecolor='k', cmap=plt.cm.Set1)
plt.title(f"KNN Decision Boundary (K={best_k})")
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.show()
```

✅ Final Output
Shows confusion matrix, decision boundary, and best accuracy achieved for optimal K.

💡 Summary of Steps

| Step                     | Description                                          |
|--------------------------|------------------------------------------------------|
| 1. Import                | Load and normalize the Iris dataset                  |
| 2. Train-Evaluate KNN    | Try K from 1 to 10 and record accuracy               |
| 3. Evaluate Best Model   | Confusion Matrix and Accuracy                        |
| 4. Decision Visualization| Plot decision boundary for selected features         |

👩‍💻 Tools Used
- Python
- Scikit-learn
- Matplotlib
- Numpy
- Google Colab

📎 Notes
This example is ideal for:
- 🚀 ML beginners to understand classification visually
- 📊 Model evaluation and hyperparameter tuning
- 🌸 Classifying flower species using sepal features
