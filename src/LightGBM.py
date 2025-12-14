import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from scipy.stats import uniform, randint
from hyperopt import hp, STATUS_OK, fmin, tpe
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns

def LightGBM_Model(X_train, X_test, y_train, y_test, class_labels):

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.fit_transform(y_test)

    space_lgbm = {
        "learning_rate": hp.loguniform("learning_rate", -3, -1),
        "num_leaves": hp.quniform("num_leaves", 15, 60, 1),
        "max_depth": hp.quniform("max_depth", 3, 10, 1),
        "n_estimators": hp.quniform("n_estimators", 200, 500, 100),
        "subsample": hp.uniform("subsample", 0.5, 1.0),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1.0),
        "lambda_l2": hp.loguniform("lambda_l2", -4, -1),
        "lambda_l1": hp.loguniform("lambda_l1", -4, -1),
    }

    def objective_lgbm(params):
    
        params["num_leaves"] = int(params["num_leaves"])
        params["max_depth"] = int(params["max_depth"])
        params["n_estimators"] = int(params["n_estimators"])

        model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=len(class_labels),
            random_state=42,
            n_jobs=-1,
            class_weight='balanced',
            **params
        )
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_weighted", n_jobs=-1)

        mean_score = scores.mean()
        
        return {'loss': -mean_score, 'status': STATUS_OK}
    

    import warnings
    warnings.filterwarnings('ignore')

    best_params = fmin(fn=objective_lgbm, space=space_lgbm, algo=tpe.suggest, max_evals=100)

    integer_keys = ["num_leaves", "max_depth", "n_estimators"]
    for k in integer_keys:
        if k in best_params:
            best_params[k] = int(best_params[k])

    float_keys = ["learning_rate", "subsample", "colsample_bytree", "lambda_l2", "lambda_l1"]
    for k in float_keys:
        if k in best_params:
            best_params[k] = float(best_params[k])

    print("\n--- Optimized LightGBM Hyperparameters ---")
    for k, v in best_params.items():
        print(f"'{k}': {v},")
    print("------------------------------------------")


    lgbm_model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(class_labels),
        verbose = 0,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced',
        **best_params
    )

    lgbm_model.fit(X_train, y_train)
    y_pred = lgbm_model.predict(X_test)

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
        cmap='Blues', 
        cbar=True,
        xticklabels=class_labels, 
        yticklabels=class_labels
    )
    plt.title("Confusion Matrix for LightGBM Classification Model")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    return lgbm_model