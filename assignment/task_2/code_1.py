#%%
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, GridSearchCV
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
from itertools import product


## Data import osv.

df = pd.read_csv("HR_data.csv")

## Whats Unnammed??
if "Unnamed: 0" in df.columns:
     df = df.drop(columns=["Unnamed: 0"])


## Frustration delt op i 3

cuts = [0,2,5,10]
labels = ["Low", "Medium", "High"]


y = pd.cut(df["Frustrated"], bins=cuts, include_lowest=True, labels=labels).astype(str)
X = df[['HR_Mean', 'HR_Median', 'HR_std', 'HR_Min', 'HR_Max', 'HR_AUC','Round', 'Phase']]

## CV settings
cv_outer = df["Cohort"]
cv_inner = df["Individual"]


#%% Jeg preprocesser data

numerical_data = Pipeline(steps=[('scaler', StandardScaler())])
categorical_data = Pipeline(steps=[('ohe', OneHotEncoder(handle_unknown="ignore"))])

preprocess = ColumnTransformer(
     transformers=[
          ('numerical', numerical_data,['HR_Mean', 'HR_Median', 'HR_std', 'HR_Min', 'HR_Max', 'HR_AUC']),
          ('categorical', categorical_data, ['Round', 'Phase'])
     ]
)

#%% Hvor mange er der af hver af de ny klasse.
summary = (
    y.value_counts()                        
     .to_frame("count")                     
     .assign(percent=lambda s:                
             (s["count"] / len(y) * 100).round(1))
)

#%% Modeller, Pipelines og Gridsearch parametrene


## Gridsearch parametrene

grid_rf = {
     'clf__n_estimators': [300, 500],
     'clf__max_depth': [None, 10, 20],
     'clf__min_samples_leaf': [1, 3, 5],
}

grid_lr = {
     'clf__C': np.logspace(-3,3,7)
}


lr_pipe = Pipeline(steps=[
     ('pre', preprocess),
     ('clf', LogisticRegression(max_iter=2000, class_weight='balanced'))
     ])

rf_pipe = Pipeline(steps=[
     ('pre',preprocess),
     ('clf',RandomForestClassifier(class_weight='balanced_subsample', random_state=43))
     ])


models = {
     'lr': (lr_pipe, grid_lr),
     'rf': (rf_pipe, grid_rf),
             }


#%%
## CV

logo = LeaveOneGroupOut()
outer_cv_results = {key: [] for key in models}
all_true  = {m: [] for m in models}
all_pred  = {m: [] for m in models}
rf_importances = []


for outer_fold, (train_idx, test_idx) in enumerate(logo.split(X,y, groups=cv_outer), 1):
     print(f'\nOuter fold {outer_fold} - test cohort: {cv_outer.iloc[test_idx].unique()[0]}')
     X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
     y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
     cv_inner_train = cv_inner.iloc[train_idx]

     inner_cv = GroupKFold(n_splits=3)

     for key, (pipe, param_grid) in models.items():
          gs = GridSearchCV(pipe, param_grid, cv=inner_cv, scoring='f1_macro',n_jobs=1,verbose=0)
          gs.fit(X_train, y_train, groups=cv_inner_train)
          y_pred = gs.predict(X_test)
          score = f1_score(y_test, y_pred, average='macro')
          outer_cv_results[key].append(score)
          
          all_true[key].extend(y_test.tolist())
          all_pred[key].extend(y_pred.tolist())
          print(f"  {key.upper():>2} – macro F1: {score:.3f}  (best: {gs.best_params_})")
          
          if key == "rf":
            rf_importances.append(gs.best_estimator_.named_steps["clf"].feature_importances_)

print("\n=== Summary statistics ===")
outer_cv_results = {k: np.asarray(v) for k, v in outer_cv_results.items()}
for key, scores in outer_cv_results.items():
    print(f"{key.upper():>2}: {scores.mean():.3f} ± {scores.std():.3f}")

stat, p = wilcoxon(outer_cv_results["lr"], outer_cv_results["rf"], alternative="less")
print(f"Wilcoxon signed‑rank test (LR < RF): statistic={stat:.3f}, p={p:.3f}")

# Save tables
pd.DataFrame(outer_cv_results).to_csv("fold_macroF1.csv", index=False)

rep_lr = classification_report(all_true["lr"], all_pred["lr"], labels=labels,
                               output_dict=True, zero_division=0)
rep_rf = classification_report(all_true["rf"], all_pred["rf"], labels=labels,
                               output_dict=True, zero_division=0)
(pd.DataFrame(rep_lr).T.add_prefix("LR_")
   .join(pd.DataFrame(rep_rf).T.add_prefix("RF_"))) \
   .to_csv("classification_report.csv")

plt.figure(figsize=(4,4))
plt.boxplot([outer_cv_results["lr"], outer_cv_results["rf"]], tick_labels=["LR", "RF"])
plt.ylabel("Macro F1")
plt.title("Macro F1 across LOCO folds")
plt.tight_layout()
plt.savefig("boxplot_macroF1.png", dpi=300)
plt.close()

for key in models:
    cm = confusion_matrix(all_true[key], all_pred[key], labels=labels)
    plt.figure(figsize=(4,4))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(f"Confusion – {key.upper()}")
    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], "d"), ha="center", va="center")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(f"confusion_{key}.png", dpi=300)
    plt.close()

if rf_importances:
    preprocess_fitted = preprocess.fit(X) 
    feat_names = preprocess_fitted.get_feature_names_out()

    mean_imp = np.mean(np.vstack(rf_importances), axis=0)
    idx = np.argsort(mean_imp)[::-1][:10]

    plt.figure(figsize=(6,4))
    plt.barh(range(len(idx)), mean_imp[idx][::-1])
    plt.yticks(range(len(idx)), feat_names[idx][::-1])
    plt.xlabel("Mean importance")
    plt.title("Random Forest – top 10 features")
    plt.tight_layout()
    plt.savefig("rf_feature_importance.png", dpi=300)
    plt.close()





# %%
