from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from scipy.stats import uniform, randint
from hyperopt import hp, STATUS_OK, fmin, tpe, space_eval
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def Cat_Boost_Model(X_train, X_test, y_train, y_test, class_labels):
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.fit_transform(y_test)

    space_catboost = {
        "depth": hp.quniform("depth", 5, 12, 1),
        "iterations": hp.quniform("iterations", 300, 1000, 100),
        "learning_rate": hp.loguniform("learning_rate", np.log(0.005), np.log(0.3)),
        "l2_leaf_reg": hp.loguniform("l2_leaf_reg", np.log(1), np.log(15)),
        "bagging_temperature": hp.uniform("bagging_temperature", 0.5, 2.0),
        "random_strength": hp.uniform("random_strength", 0.1, 1.0), 
        "border_count": hp.choice("border_count", [32, 64, 128, 255])
    }

    def objective_catboost(params):
        params["depth"] = int(params["depth"])
        params["iterations"] = int(params["iterations"])

        model = CatBoostClassifier(
            objective = "MultiClass",
            eval_metric = "MultiClass",
            random_seed = 42,
            verbose = 0,
            thread_count=-1,
            **params
        )

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_weighted", n_jobs=-1)

        mean_score = scores.mean()
        
        return {'loss': -mean_score, 'status': STATUS_OK}
    
    import warnings
    warnings.filterwarnings('ignore')

    best_params = fmin(fn=objective_catboost, space=space_catboost, algo=tpe.suggest, max_evals=100)

    final_params = space_eval(space_catboost, best_params)

    final_params["depth"] = int(final_params["depth"])
    final_params["iterations"] = int(final_params["iterations"])
    final_params["border_count"] = int(final_params["border_count"])

    print("\n--- Optimized CatBoost Hyperparameters ---")
    for k, v in best_params.items():
        print(f"'{k}': {v},")
    print("------------------------------------------")


    catboost_model = CatBoostClassifier(
        objective="MultiClass",
        eval_metric="MultiClass",
        random_seed=42,
        verbose=0,
        thread_count=-1,
        **final_params
    )

    catboost_model.fit(X_train, y_train)
    y_pred = catboost_model.predict(X_test)

    y_test = le.inverse_transform(y_test)
    y_pred = le.inverse_transform(y_pred)

    print(classification_report(y_test, y_pred))

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    cm = confusion_matrix(y_test, y_pred, labels=class_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True,     
        fmt='d',        
        cmap='plasma', 
        cbar=True,
        xticklabels=class_labels, 
        yticklabels=class_labels
    )
    plt.title("Confusion Matrix for CatBoost Classification Model")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    return catboost_model