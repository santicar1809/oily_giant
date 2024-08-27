import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split

def bootstraping(predict,income_per_barrel,mean_income_per_well):
    state=np.random.RandomState(12345)
    mean_values=[]
    values_income=[]
    for i in range(1000):
        pred_subsample=predict.sample(n=500,replace=True,random_state=state)
        incomes=pred_subsample*income_per_barrel-mean_income_per_well
        mean_boot=incomes.mean()
        mean_values.append(mean_boot)
        values_income.append(incomes.quantile(0.99))
        values_income_df=pd.Series(values_income)
        mean_value=values_income_df.mean()
    return values_income_df,mean_value

def confiance(data):
    lower=data.quantile(0.05)
    upper=data.quantile(0.95)
    interval=str(lower)+' - '+ str(upper)
    return interval

def lose_risk(data):
    looses_total=0
    looses=data[data<0].sum()
    looses_total+=looses
    risk_prob=looses/data.shape[0]
    return risk_prob    

def incomes(pred,income_per_barrel,mean_income_per_well,budget,count):
    sorted_wells=pred.sort_values(ascending=False)
    top=sorted_wells.head(count)
    mean_barrels=top.mean()
    total_incomes=sum(top*income_per_barrel-mean_income_per_well)
    revenue_per_region = total_incomes-mean_income_per_well
    roi = 100*(total_incomes-budget)/budget
    values , bootstraping_mean = bootstraping(top,income_per_barrel,mean_income_per_well)
    interval=confiance(values)
    risk = lose_risk(values)
    return total_incomes,mean_barrels,revenue_per_region,roi,bootstraping_mean,interval,risk

def rev_report(data):

    model = joblib.load('./files/modeling_output/model_fit/best_random_Linrego.joblib')
    seed=12345
    budget=100000000
    wells=200
    income_per_barrel=4500
    mean_income_per_well=500000
    metrics = []
    for df in data:
        features=df.drop(['id','product'],axis=1)
        target=df['product']
        features_train,features_valid,target_train,target_valid=train_test_split(features,target,test_size=0.25,random_state=seed)
        pred = model.predict(features_valid)
        pred=pd.Series(pred,index=target_valid.index)
        total_incomes,mean_barrels,revenue_per_region,roi,bootstraping_mean,interval,risk = incomes(pred,income_per_barrel,mean_income_per_well,budget,wells)
        metrics.append([total_incomes,mean_barrels,revenue_per_region,roi,bootstraping_mean,interval,risk])
    report_df=pd.DataFrame(metrics,columns=['total_revenue', 'miles_mean_barrels','region_revenue','roi','beneficio_bootstraping','interval','lose_risk'],
                            index=['Region_0','Region_1','Region_2'])
    report_df.to_csv('./files/modeling_output/reports/revenue_report.csv')
    