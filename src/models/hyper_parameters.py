from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
import numpy as np

## Logistic Regression Model
def all_models():
    '''This function will host all the model parameters, can be used to iterate the
    grid search '''

    dummie_pipeline=Pipeline([
        ('dummie',DummyRegressor())])

    dummie_params={}
    
    dummie=['dummie',dummie_pipeline,dummie_params]

    lr_pipeline = Pipeline([
        ('Linreg', LinearRegression())
    ])

    lr_param_grid = {
                    'dummie__fit_intercept': [True, False],
                    'dummie__copy_X': [True, False],
                    'dummie__n_jobs': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
                    'dummie__positive': [True, False]}

    lr = ['Linreg',lr_pipeline,lr_param_grid]

    lro_param_grid={}
    lro=['Linrego',lr_pipeline,lro_param_grid]
    
    models = [lr,lro,dummie] #Activate to run all the models
    return models
