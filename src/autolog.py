import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import dagshub
# Comment out dagshub.init() to use local MLflow tracking
# dagshub.init(repo_owner='aryanrajput8962', repo_name='YT-MLOPS-Experiment-with-MLFLOW', mlflow=True)

# Use local tracking URI for local MLflow UI
mlflow.set_tracking_uri("mlruns")

# The tracking URI is automatically set by dagshub.init() above
# import mlflow

# with mlflow.start_run():
#   mlflow.log_param('parameter name', 'value')
#   mlflow.log_metric('metric name', 1)
# Load Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Define the params for RF model
max_depth_1 = 10
n_estimators_1 = 5

max_depth_2 = 8
n_estimators_2 = 10

# ============== EXPERIMENT 1 ==============
mlflow.autolog()
mlflow.set_experiment('YT-MLOPS-Exp1')


with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth_1, n_estimators=n_estimators_1, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # mlflow.log_metric('accuracy', accuracy)
    # mlflow.log_param('max_depth', max_depth_1)
    # mlflow.log_param('n_estimators', n_estimators_1)

    # Creating a confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix - Exp1')

    # save plot
    plt.savefig("Confusion-matrix-exp1.png")

    # # log artifacts using mlflow
    # mlflow.log_artifact("Confusion-matrix-exp1.png")
    # mlflow.log_artifact(__file__)

    # tags
    mlflow.set_tags({"Author": 'Aryan', "Project": "Wine Classification"})

    # Log the model
    # mlflow.sklearn.log_model(rf, "Random-Forest-Model")

    print(f"Experiment 1 Accuracy: {accuracy}")

# ============== EXPERIMENT 2 ==============
mlflow.autolog()
mlflow.set_experiment('YT-MLOPS-Exp2')

with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth_2, n_estimators=n_estimators_2, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # mlflow.log_metric('accuracy', accuracy)
    # mlflow.log_param('max_depth', max_depth_2)
    # mlflow.log_param('n_estimators', n_estimators_2)

    # Creating a confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix - Exp2')

    # save plot
    plt.savefig("Confusion-matrix-exp2.png")

    # # log artifacts using mlflow
    # mlflow.log_artifact("Confusion-matrix-exp2.png")
    # mlflow.log_artifact(__file__)

    # tags
    mlflow.set_tags({"Author": 'Aryan', "Project": "Wine Classification"})

    # Log the model
    # mlflow.sklearn.log_model(rf, "Random-Forest-Model")

    print(f"Experiment 2 Accuracy: {accuracy}") 
    
