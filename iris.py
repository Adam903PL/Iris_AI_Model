import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

print("10%")
df = pd.read_csv("iris.csv")

X = df.drop('species', axis=1)
y = df['species']

print("20%")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print("30%")
plt.figure(figsize=(12, 10))


plt.subplot(2, 2, 1)
sns.histplot(data=df, x='sepal_length', hue='species', kde=True)
plt.title('Sepal Length Distribution')

plt.subplot(2, 2, 2)
sns.histplot(data=df, x='sepal_width', hue='species', kde=True)
plt.title('Sepal Width Distribution')

plt.subplot(2, 2, 3)
sns.histplot(data=df, x='petal_length', hue='species', kde=True)
plt.title('Petal Length Distribution')

plt.subplot(2, 2, 4)
sns.histplot(data=df, x='petal_width', hue='species', kde=True)
plt.title('Petal Width Distribution')

plt.tight_layout()
plt.savefig('feature_hist.png')

print("40%")
plt.figure(figsize=(12, 10))
sns.pairplot(df, hue='species')
plt.savefig('pairplot.png')

print("50%")
plt.figure(figsize=(10, 8))
correlation = df.drop('species', axis=1).corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('corr_matrix.png')

print("60%")
models = [
    ('LogReg', LogisticRegression(max_iter=200, random_state=42)),
    ('KNN', KNeighborsClassifier(n_neighbors=5)),
    ('SVM', SVC(random_state=42)),
    ('DT', DecisionTreeClassifier(random_state=42)),
    ('RF', RandomForestClassifier(random_state=42)),
    ('GB', GradientBoostingClassifier(random_state=42)),
    ('NB', GaussianNB())
]

results = []
names = []

print("70%")
for name, model in models:
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])

    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
    results.append(cv_scores)
    names.append(name)

print("80%")
plt.figure(figsize=(12, 6))
box = plt.boxplot(results, patch_artist=True, tick_labels=names)
for patch in box['boxes']:
    patch.set_facecolor('lightblue')
plt.title('Model Comparison - Cross Validation')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.savefig('model_comparison.png')

print("85%")
best_model = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(random_state=42))
])

best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

print("90%")
feature_importance = pd.DataFrame(
    best_model.named_steps['model'].feature_importances_,
    index=X.columns,
    columns=['importance']
).sort_values('importance', ascending=False)

top_features = feature_importance.index[:2].tolist()

print("95%")
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_train[top_features[0]], X_train[top_features[1]],
                      c=[list(y.unique()).index(label) for label in y_train],
                      cmap='viridis', edgecolor='black', s=50)
plt.scatter(X_test[top_features[0]], X_test[top_features[1]],
            c=[list(y.unique()).index(label) for label in y_test],
            cmap='viridis', edgecolor='white', s=20)
plt.xlabel(top_features[0])
plt.ylabel(top_features[1])
plt.title('Top Features Distribution')
plt.colorbar(scatter, ticks=[0, 1, 2], label='Species')
plt.tight_layout()
plt.savefig('top_features.png')

print("100%")




# Tu Poprosiłem chata by ładnie wszystko wyprintował aby było czytelne

print("RESULTS SUMMARY")
print("--------------")
print(f"Dataset shape: {df.shape}")
print(f"Species distribution: {y.value_counts().to_dict()}")
print("\nModel performance (cross-validation):")
for i, (name, _) in enumerate(models):
    print(f"{name}: mean={results[i].mean():.4f}, std={results[i].std():.4f}")

print("\nBest model (Random Forest) test results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nFeature Importance:")
print(feature_importance)