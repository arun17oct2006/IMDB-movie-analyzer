import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve
import seaborn as sns
from sklearn.metrics import confusion_matrix

df = pd.read_csv('/kaggle/input/imdbreview/IMDB Dataset.csv')

df['sentiment'] = df['sentiment'].map({"positive": 1, "negative": 0})

df = df.dropna(subset=['review', 'sentiment'])

X_train, X_val, y_train, y_val = train_test_split(
    df['review'], df['sentiment'], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)

X_train_dense = X_train_vec.toarray()
X_val_dense = X_val_vec.toarray()

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train_dense, y_train)


y_pred = model.predict(X_val_vec)
acc=accuracy_score(y_val, y_pred)
print(" Accuracy:", accuracy_score(y_val, y_pred))
print("\nClassification Report:\n", classification_report(y_val, y_pred))
cm = confusion_matrix(y_val, y_pred)
print("\nConfusion Matrix:\n", cm)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap="Reds",
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Random Forest")
plt.show()

log_reg = LogisticRegression(max_iter=1000, n_jobs=-1)
log_reg.fit(X_train_vec, y_train)

y_pred = log_reg.predict(X_val_vec)
acc1=accuracy_score(y_val, y_pred)
# Evaluation
print("Logistic Regression Accuracy:", accuracy_score(y_val, y_pred))
print("\nClassification Report:\n", classification_report(y_val, y_pred))
# Step 9: Confusion Matrix
cm1 = confusion_matrix(y_val, y_pred)

print("\nConfusion Matrix (numeric):\n", cm1)

# Plot confusion matrix (pink theme)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap="Greens",
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Logistic Regression ")
plt.show()
if acc>acc1:
    print("Random forest have more accuracy")
else:
    print("Logistic regression have more accuracy")


# Function to plot learning curves
def plot_learning_curve(model, X, y, title="Learning Curve"):
    plt.figure(figsize=(8,6))

    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y,
        cv=5,
        scoring="accuracy",
        train_sizes=np.linspace(0.1, 1.0, 8),
        n_jobs=-1
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std  = np.std(train_scores, axis=1)
    val_mean   = np.mean(val_scores, axis=1)
    val_std    = np.std(val_scores, axis=1)

    plt.plot(train_sizes, train_mean, 'o-', color="blue", label="Training accuracy")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="blue")

    plt.plot(train_sizes, val_mean, 'o-', color="green", label="Validation accuracy")
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color="green")

    plt.title(title)
    plt.xlabel("Training set size")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()

plot_learning_curve(RandomForestClassifier(n_estimators=200, random_state=42),
                    X_train_vec, y_train,
                    title="Random Forest Learning Curve")

plot_learning_curve(LogisticRegression(max_iter=1000, n_jobs=-1),
                    X_train_vec, y_train,
                    title="Logistic Regression Learning Curve")


print("\n--- IMDB Movie Sentiment Prediction ---")
user_review = input("Enter a movie review: ")

user_vec = vectorizer.transform([user_review])

pred_rf = model.predict(user_vec)[0]
print(f"Random Forest Prediction: {'Positive' if pred_rf==1 else 'Negative'}")

pred_lr = log_reg.predict(user_vec)[0]
print(f"Logistic Regression Prediction: {'Positive' if pred_lr==1 else 'Negative'}")


