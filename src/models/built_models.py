import pandas as pd 
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV,RandomizedSearchCV
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from src.models.hyper_parameters import all_models
import joblib

def iterative_modeling(data):
    '''This function will bring the hyper parameters from all_model() 
    and wil create a complete report of the best model, estimator, 
    score and validation score'''
    
    models = all_models() 
    
    output_path = './files/modeling_output/model_fit/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    
    results = []
    data_name=['data_0','data_1','data_2']
    # Iterating the models
    for df,i in zip(data,data_name):
        results = []    
        for model in models:
            best_estimator, best_score, rmse_val,r2_val,eam_val= model_structure(df, model[1], model[2])
            results.append([model[0],best_estimator,best_score, rmse_val,r2_val,eam_val])      
            # Guardamos el modelo
            joblib.dump(best_estimator,output_path +f'best_random_{model[0]}.joblib')
        results_df = pd.DataFrame(results, columns=['model','best_estimator','best_train_score','rmse_score','r2_score','eam_val'])

        results_df.to_csv(f'./files/modeling_output/reports/model_report_{i}.csv',index=False)

    return 


def model_structure(data, pipeline, param_grid):
    '''This function will host the structure to run all the models, splitting the
    dataset, oversampling the data and returning the scores'''
    seed=12345
    features=data.drop(['id','product'],axis=1)
    target=data['product']
    features_train,features_valid,target_train,target_valid=train_test_split(features,target,test_size=0.25,random_state=seed)   
    # Training the model
    gs = RandomizedSearchCV(pipeline, param_grid, cv=2, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
    gs.fit(features_train,target_train)

    # Scores
    best_score = gs.best_score_
    best_estimator = gs.best_estimator_
    rmse_val,r2_val,eam_val = eval_model(best_estimator,features_valid,target_valid)
    
    results = best_estimator, best_score,rmse_val,r2_val,eam_val
    return results
    
def eval_model(best,features_valid,target_valid):
    random_prediction=best.predict(features_valid)
    ecm_random=mean_squared_error(target_valid,random_prediction)
    recm_random=ecm_random**0.5
    r2_random=r2_score(target_valid,random_prediction)
    eam_random=mean_absolute_error(target_valid,random_prediction)
    return recm_random,r2_random,eam_random