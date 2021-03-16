
# Loading libraries

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
# import pyodbc
import numpy as np
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
# from sklearn.externals import joblib
# import sklearn.external.joblib as extjoblib
import joblib
from pandas import ExcelWriter
import configparser
# from sqlalchemy import create_engine
import urllib

# Reading configuration file

config = configparser.RawConfigParser()

try:
    config.read('Configuration.txt')
except Exception as e:
    print(str(e))
try:
    '''
    server = config.get('SQL_server', 'server')
    DB = config.get('SQL_server', 'DB')
    ID = config.get('SQL_server', 'ID')
    PWD = config.get('SQL_server', 'PWD')
    query = config.get('SQL_server', 'sql_query')
    '''

    model_path = config.get('Paths', 'model_path')
    metric_path = config.get('Paths', 'metric_path')
    le_path = config.get('Paths', 'le_path')
    input_path = config.get('Paths', 'input_path')
    output = config.get('Paths', 'output_path')


except Exception as e:
    print('Could not read configuration file. {}'.format(str(e)))


# conecting to database server retriving data

print("Connecting to data")

'''
con = pyodbc.connect(r'Driver={SQL Server};'
                                'Server='+server+';'
                                'Database='+DB+';'
                                'uid='+ID+';pwd='+PWD)

INC_df = pd.read_sql_query(query, con)
'''

INC_df = pd.read_csv(r'A:\MLOps\Incident prediction\IncidentPredictionMLOps_v1\input\data_pre.csv')

# Training function

def modelbuild(df, DS=""):

    # Label encoding
    
    df1 = df.iloc[:, 1:]
    le = LabelEncoder()
    for i in df1.columns:
        if (df1[i].dtype == 'object'):
            df1[i] = le.fit_transform(df1[i])
            joblib.dump(le, le_path + i + DS + '.pkl')

    y = df1.No_of_incidents
    x = df1.drop('No_of_incidents', axis=1)

    # Model building

    models = {"LinearRegression": LinearRegression(), "DecisionTreeRegressor": DecisionTreeRegressor(),
              "RandomForestRegressor": RandomForestRegressor(),
              'KNeighborsRegressor': KNeighborsRegressor(n_neighbors=4)}

    model_name = ['MLR', 'DTR', 'RFR', 'KNNR']
    r2_list = []
    rmse_list = []


    for i, j in zip(models.values(), model_name):
        mod = i.fit(x, y)
        joblib.dump(mod, model_path+'1_' + j +'_'+ DS + '.pkl')
        mod_P = mod.predict(x).astype('int64')

        # Metric calculation

        mod_r2 = r2_score(y, mod_P)
        mod_rmse = np.sqrt(mean_squared_error(y, mod_P))

        r2_list.append(mod_r2)
        rmse_list.append(mod_rmse)

    df_met = pd.DataFrame(list(zip(model_name, r2_list, rmse_list)), columns=['Model name', 'R-square', 'RMSE'])

    return df_met

# Removing outliers

train_df = INC_df[(INC_df['No_of_incidents']<50)].drop('discovery source',axis = 1)


# Training models

print("Model building started...")

metrics = modelbuild(train_df, DS = 'User Created')

print("Models has been trained.")




