import pandas as pd
import os
import re
import numpy as np
import matplotlib.pyplot as plt

def to_snake_case(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    s1 = s1.replace(' ','_')
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def columns_transformer(data):
    #Pasamos las columnas al modo snake_case
    columns=data.columns
    new_cols=[]
    for i in columns:
        i=to_snake_case(i)
        new_cols.append(i)
    data.columns=new_cols
    print(data.columns)
    return data

def nan_values(data):
    # Tratamiento de ausentes
    for column in data.columns:   
        if data[column].isna().sum()/data.shape[0] < 0.15:
            mode=data[column].mode()[0]
            data[column].fillna(value=mode,inplace=True)
        elif data[column].isna().sum()/data.shape[0] > 0.15:
            data.dropna(inplace=True)
    return data

def duplicated_values(data):
    # Tratamiento de duplicados
    if data.duplicated().sum() > 0:
            data.drop_duplicates()
    return data

def preprocess_data(data):
    '''This function will clean the data by setting removing duplicates, 
    formatting the column types, names and removing incoherent data. The datasets
    will be merged in one joined by the CustomerID''' 
        
    data_0=data[0]
    data_1=data[1]
    data_2=data[2]
    
    # Pasamos columnas a formato snake_case
        
    data_0 = columns_transformer(data_0)
    data_1 = columns_transformer(data_1)
    data_2 = columns_transformer(data_2)
    
    # Ausentes
    
    data_0=nan_values(data_0)
    data_1=nan_values(data_1)
    data_2=nan_values(data_2)
    
    # Boxplots
    data=[data_0,data_1,data_2]
    data_name=['data_0','data_1','data_2']
    for df,i in zip(data,data_name):
        fig,ax=plt.subplots()
        ax.boxplot(df['product'])
        ax.set_title('Product')   
        fig.savefig(f'./files/modeling_output/figures/box_{i}_product')
    
    # Duplicates
    
    data_0=duplicated_values(data_0)
    data_1=duplicated_values(data_1)
    data_2=duplicated_values(data_2)
    path = './files/datasets/intermediate/'

    data_0.to_csv(path+'preprocessed_data_0.csv', index=False)
    data_1.to_csv(path+'preprocessed_data_1.csv', index=False)
    data_2.to_csv(path+'preprocessed_data_2.csv', index=False)
    
    return data_0,data_1,data_2