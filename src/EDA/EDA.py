import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
import plotly.express as px
import sys
import os
from sklearn.metrics import mean_squared_error

def eda_report(data):
    '''Te EDA report will create some files to analyze the in deep the variables of the table.
    The elements will be divided by categoric and numeric and some extra info will printed'''
    
    data_0=data[0]
    data_1=data[1]
    data_2=data[2]
    data=[data_0,data_1,data_2]
    data_name=['data_0','data_1','data_2']
    for df,i in zip(data,data_name):
        describe_result=df.describe()

        eda_path = './files/modeling_output/figures/'
        reports_path='./files/modeling_output/reports/'
        if not os.path.exists(reports_path):
            os.makedirs(reports_path)

        if not os.path.exists(eda_path):
            os.makedirs(eda_path)

        # Exporting the file
        with open(reports_path+f'describe_{i}.txt', 'w') as f:
            f.write(describe_result.to_string())

        # Exporting general info
        with open(reports_path+f'info_{i}.txt','w') as f:
            sys.stdout = f
            df.info()
            sys.stdout = sys.__stdout__

        fig , ax  = plt.subplots()
        ax.hist(df['product'])
        fig.savefig(eda_path+f'fig_{i}.png')
