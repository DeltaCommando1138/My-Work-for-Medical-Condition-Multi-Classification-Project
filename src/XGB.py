from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from hyperopt import fmin, tpe, hp, STATUS_OK
from sklearn.model_selection import cross_val_score, StratifiedKFold


def XGB_Model(X_train, X_test, y_train, y_test, class_labels):

    #Integer Encoding for XGB
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.transform(y_test)

    space = {
        "learning_rate": hp.loguniform("learning_rate", -3, -1),
        "max_depth": hp.quniform("max_depth", 3, 8, 1),
        "n_estimators": hp.quniform("n_estimators", 200, 500, 100),
        "subsample": hp.uniform("subsample", 0.5, 1.0),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1.0),
        "reg_lambda": hp.loguniform("reg_lambda", -4, -1)
    }

    def objective(params):
        params["max_depth"] = int(params["max_depth"])
        params["n_estimators"] = int(params["n_estimators"])

        model = XGBClassifier(
            objective="multi:softprob",
            num_class=len(class_labels),
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
            **params
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)

        mean_score = scores.mean()
        return {'loss': -mean_score, 'status': STATUS_OK}
    
    best_params = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100)
    for k in ["max_depth", "n_estimators"]:
        best_params[k] = int(best_params[k])
    for k in ["learning_rate", "subsample", "colsample_bytree", "reg_lambda"]:
        best_params[k] = float(best_params[k])

    xgb_model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(class_labels),
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        **best_params)

    #Train and Predict
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)

    #Transform back to the original labels for plotting 
    y_test = encoder.inverse_transform(y_test)
    y_pred = encoder.inverse_transform(y_pred)    

    #Report
    accuracy = accuracy_score(y_test, y_pred)
    print(classification_report(y_test, y_pred, target_names=class_labels))
    print(f"Accuracy: {accuracy}")

    #Plot confusion matrix
    plt.figure(figsize=(12, 12))
    cm = confusion_matrix(y_test, y_pred, labels=class_labels)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels)
    plt.title("Confustion Matrix for XGB Classifier")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

    return xgb_model

