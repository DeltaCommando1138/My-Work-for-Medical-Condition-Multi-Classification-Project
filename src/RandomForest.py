from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import cross_val_score,StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from hyperopt import fmin,tpe,hp,STATUS_OK
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def RF_Model(X_train,X_test,y_train,y_test,class_labels):

    le = LabelEncoder()
    le.fit(class_labels)

    if np.issubdtype(y_train.dtype, np.number):
        y_train_encoded = y_train
    else:
        y_train_encoded = le.transform(y_train)

    if np.issubdtype(y_test.dtype, np.number):
        y_test_labels = le.inverse_transform(y_test)
    else:
        y_test_labels = y_test


    space={
        "n_estimators":hp.quniform("n_estimators",100,400,50),
        "max_depth":hp.quniform("max_depth",5,20,1),
        "min_samples_split":hp.quniform("min_samples_split",2,10,1),
        "min_samples_leaf":hp.quniform("min_samples_leaf",1,5,1),
        "max_features":hp.choice("max_features",["sqrt","log2",None])
    }


    def objective(params):
        params["n_estimators"]=int(params["n_estimators"])
        params["max_depth"]=int(params["max_depth"])
        params["min_samples_split"]=int(params["min_samples_split"])
        params["min_samples_leaf"]=int(params["min_samples_leaf"])
        model=RandomForestClassifier(random_state=42,n_jobs=-1,**params)
        cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
        scores=cross_val_score(model,X_train,y_train_encoded,cv=cv,scoring="accuracy",n_jobs=-1)
        return {'loss':-scores.mean(),'status':STATUS_OK}


    best_params=fmin(fn=objective,space=space,algo=tpe.suggest,max_evals=100)


    for k in ["n_estimators","max_depth","min_samples_split","min_samples_leaf"]:
        best_params[k]=int(best_params[k])
    max_features_options=["sqrt","log2",None]
    best_params["max_features"]=max_features_options[int(best_params["max_features"])]


    rf_model=RandomForestClassifier(random_state=42,n_jobs=-1,**best_params)
    rf_model.fit(X_train,y_train_encoded)

    y_pred_encoded = rf_model.predict(X_test)
    y_pred_labels = le.inverse_transform(y_pred_encoded)

    accuracy = accuracy_score(y_test_labels, y_pred_labels)


    print("Classification Report (Random Forest):")
    print(classification_report(y_test_labels,y_pred_labels,target_names=class_labels))
    print(f"Accuracy: {accuracy}")


    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test_labels, y_pred_labels, labels=class_labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=class_labels, yticklabels=class_labels)
    plt.title("Confusion Matrix - Random Forest")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()


    return rf_model
