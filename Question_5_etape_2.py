import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import xgboost 
from comet_ml import Experiment
import seaborn as sns
import os
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn import preprocessing,svm, datasets
from sklearn.metrics import roc_curve, auc,roc_auc_score,accuracy_score
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.calibration import CalibrationDisplay

COMET_PROJECT_NAME = "ift6758-project"
COMET_WORKSPACE = "ift6758-22-milestone-2"

NET_ABSOLUTE_COORD_X = 89
NET_COORD_Y = 0

sns.set(style="darkgrid")





"""
Creates a comet experiment with the right project configuration. 
It will get the api key from this environment variable: COMET_API_KEY
"""
comet_api_key = 'jJEs8xkFH63p4ubkYK2JRxJu1'
experiment = Experiment(
    log_code=True,
    api_key=comet_api_key,
    project_name=COMET_PROJECT_NAME,
    workspace=COMET_WORKSPACE,
)
    


from generic_util import plot_goals_rate_pdf, \
                            plot_goals_rate_cdf, \
                            scale_features


def train(X_train, X_valid, y_train, y_valid,p):
    if p == None : 
        model_xgboost = xgboost.XGBClassifier()
        eval_set = [(X_valid, y_valid)]
    if p != None:
        eval_set = [(X_train, y_train),(X_valid, y_valid)]
        
    
        model_xgboost = xgboost.XGBClassifier(learning_rate=p[0],
                                          max_depth=p[1],
                                          n_estimators=p[2],
                                          subsample=p[3],
                                          colsample_bytree=p[4],
                                          eval_metric=p[5],
                                          verbosity=p[6],
                                          use_label_encoder=p[7])
        
        
        
    
    model_xgboost.fit(X_train,
                  y_train,
                  eval_set=eval_set,
                  verbose=True)
    
    y_pred = model_xgboost.predict(X_valid)
    acc = accuracy_score(y_valid, y_pred)
    
    y_probas = model_xgboost.predict_proba(X_valid)
    y_probas = y_probas[:,1]
    
    return y_pred, acc, y_probas, model_xgboost



def Graphs_Question1(X,y,feats,parameters):
    
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
    
        
    y_pred, acc, y_probas, model_xgboost = train(X_train, X_valid, y_train, y_valid ,parameters)
    
    
    # a)
    fpr, tpr, thresholds = roc_curve(y_valid, y_probas)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr,label=feats)
    
    plt.legend()
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    # b)
    
    pdf = []
    x = np.arange(100)
    for i in range(100):
        threshold = np.percentile(y_probas, i)
        goals = len([y_prob for y_prob, y in zip(y_probas, y_valid) if y_prob >= threshold and y == 1])
        non_goals = len([y_prob for y_prob, y in zip(y_probas, y_valid) if y_prob >= threshold and y == 0])
        pdf.append(100*(goals / (goals + non_goals)))
   
    
    plt.plot(x, pdf,label=feats)

    plt.ylim(0, 100)
    plt.legend()
    plt.title('Goal Rate')
    plt.ylabel('Goals / (Goals + Shoots)')
    plt.xlabel('Shot Probability Model Percentile')
    plt.show()
  
    # c)
    
    goals_tot = sum(y_valid)
    
    cdf = []
    for i in range(100):
        threshold = np.percentile(y_probas, i)
        goals = len([y_prob for y_prob, y in zip(y_probas, y_valid) if y_prob >= threshold and y == 1])
        # goals = len([y_prob for y_prob, y in zip(y_probas, y_true) if y_prob <= threshold and y == 1])
        cdf.append(100*(goals / goals_tot))
    # data['x'] = x
    # data['cdf'] = cdf
    # sns.lineplot(data=data, x=x, y=cdf)
    plt.plot(x, cdf, label=feats)

    plt.ylim(0, 100)
    plt.legend()
    plt.title('Cumulative % of goals')
    plt.ylabel('Proportion (%)')
    plt.xlabel('Shot Probability Model Percentile')
    plt.show()

    # d)
    disp = CalibrationDisplay.from_estimator(model_xgboost, X_valid, y_valid)
    plt.show()
    disp = CalibrationDisplay.from_predictions(y_valid, y_probas)
    plt.show()
    
    
    return model_xgboost








csv_path1 = './data/train-q4-3.csv'
df = pd.read_csv(csv_path1)
y = df.loc[:, 'is_goal']


#Question1
feats = ['shot_distance', 'shot_angle']
simple = df.loc[:,feats]
model_xgboost = Graphs_Question1(simple, y,feats,None)




#Question2
feats = ['goal_strength_code','goal_empty_net','event_type','is_goal','event_id','game_start_time','team_name', 'game_end_time','season','game_id','team_id','tean_name','team_tri_code','team_link','empty_net']
lbl = preprocessing.LabelEncoder()
df['shot_type'] = lbl.fit_transform(df['shot_type'].astype(str))
df['last_event_type'] = lbl.fit_transform(df['last_event_type'].astype(str))
df['shooter_name'] = lbl.fit_transform(df['shooter_name'].astype(str))
df['goalie_name'] = lbl.fit_transform(df['goalie_name'].astype(str))
df['team_rink_side_right'] = lbl.fit_transform(df['team_rink_side_right'].astype(bool))
df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
X = df.loc[:, ~df.columns.isin (feats)]
learning_rate_list = [0.02, 0.05,0.1]
max_depth_list = [ 3, 4, 5]
n_estimators_list = [1000, 2000, 3000]

params_dict = {"learning_rate": learning_rate_list,
              "max_depth": max_depth_list,
              "n_estimators": n_estimators_list}

num_combinations = 1
for v in params_dict.values(): num_combinations *= len(v) 

print(num_combinations)
params_dict
def my_roc_auc_score(model, X, y): return roc_auc_score(y, model.predict_proba(X)[:,1])

model_xgboost_hp = GridSearchCV(estimator=xgboost.XGBClassifier(subsample=0.5,
                                                                colsample_bytree=0.25,
                                                                eval_metric='auc',
                                                                use_label_encoder=False),
                                                                param_grid=params_dict,
                                                                cv=2,
                                                                scoring=my_roc_auc_score,
                                                                return_train_score=True,
                                                                verbose=4,
                              )

model_xgboost_hp.fit(X, y)  






df_cv_results = pd.DataFrame(model_xgboost_hp.cv_results_)
df_cv_results = df_cv_results[['rank_test_score','mean_test_score','mean_train_score',
                               'param_learning_rate', 'param_max_depth', 'param_n_estimators']]
df_cv_results.sort_values(by='rank_test_score', inplace=True)
df_cv_results  
# First sort by number of estimators as that would be x-axis
df_cv_results.sort_values(by='param_n_estimators', inplace=True)

# Find values of AUC for learning rate of 0.05 and different values of depth
lr_d2 = df_cv_results.loc[(df_cv_results['param_learning_rate']==0.02) & (df_cv_results['param_max_depth']==3),:]
lr_d3 = df_cv_results.loc[(df_cv_results['param_learning_rate']==0.02) & (df_cv_results['param_max_depth']==4),:]
lr_d5 = df_cv_results.loc[(df_cv_results['param_learning_rate']==0.02) & (df_cv_results['param_max_depth']==5),:]


# Let us plot now
fig, ax = plt.subplots(figsize=(10,5))
lr_d2.plot(x='param_n_estimators', y='mean_test_score', label='Depth=3', ax=ax)
lr_d3.plot(x='param_n_estimators', y='mean_test_score', label='Depth=4', ax=ax)
lr_d5.plot(x='param_n_estimators', y='mean_test_score', label='Depth=5', ax=ax)
plt.ylabel('Mean Validation AUC')
plt.title('Performance wrt # of Trees and Depth')
# First sort by number of estimators as that would be x-axis
df_cv_results.sort_values(by='param_n_estimators', inplace=True)


# Find values of AUC for learning rate of 0.05 and different values of depth
lr_t3k_d2 = df_cv_results.loc[(df_cv_results['param_n_estimators']==2000) & (df_cv_results['param_max_depth']==4),:]

# Let us plot now
fig, ax = plt.subplots(figsize=(10,5))
lr_t3k_d2.plot(x='param_learning_rate', y='mean_test_score', label='Depth=4, Trees=2000', ax=ax)
plt.ylabel('Mean Validation AUC')
plt.title('Performance wrt learning rate')



parameters = [0.02,4,2000,0.5,0.25,'auc',1,False]




model_xgboost_fin = Graphs_Question1(X, y, 'caracteristiques',parameters)
 


#Question3
evaluation_results = model_xgboost_fin.evals_result()

# Index into each key to find AUC values for training and validation data after each tree
train_auc_tree = evaluation_results['validation_0']['auc']
valid_auc_tree = evaluation_results['validation_1']['auc']


# Plotting Section
plt.figure(figsize=(15,5))

plt.plot(train_auc_tree, label='Train')
plt.plot(valid_auc_tree, label='valid')

plt.title("Train and validation AUC as number of trees increase")
plt.xlabel("Trees")
plt.ylabel("AUC")
plt.legend(loc='lower right')
plt.show()



df_var_imp = pd.DataFrame({"Variable": X.columns,
                           "Importance": model_xgboost_fin.feature_importances_}) \
                        .sort_values(by='Importance', ascending=False)
df_var_imp

explainer = shap.Explainer(model_xgboost_fin)
shap_values = explainer(X)
shap.plots.bar(shap_values, max_display=23)


exclude = ['team_rink_side_right','rebond', 'team_home']

X = X.loc[:, ~X.columns.isin (exclude)]

model_xgboost_fin = Graphs_Question1(X, y, 'caracteristiques',parameters)

    