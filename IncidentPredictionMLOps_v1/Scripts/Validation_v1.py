

# importing libraries

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import pyodbc
import numpy as np
from sklearn.externals import joblib
import configparser
import datetime
from sqlalchemy import create_engine
import urllib
from Data_preparation import *

# Reading configuration file

config = configparser.RawConfigParser()

try:
    config.read('Configuration.txt') 
except Exception as e:
    print(str(e))
try:

	server = config.get('SQL_server', 'server')
	DB = config.get('SQL_server', 'DB')
	ID = config.get('SQL_server', 'ID')
	PWD = config.get('SQL_server', 'PWD')

	model_path = config.get('Paths', 'model_path')
	le_path = config.get('Paths', 'le_path')

	date = config.get('Date', 'validation_date')
	val_range = config.get('Date', 'validation_range')

except Exception as e:

    print('Could not read configuration file. {}'.format(str(e)))

# conecting to database server retriving data

print("Connecting to DB server")

'''
con = pyodbc.connect(r'Driver={SQL Server};'
                                'Server='+server+';'
                                'Database='+DB+';'
                                'uid='+ID+';pwd='+PWD)

query = "select ACTUALSTART,Source,CustomerSeverity,DEVICE_TYPE,COMMODITYGROUP,PROVINCE \
from IMIS_FACT_CurrentIncidentDetails_POC \
where Source='User created' and COMMODITYGROUP in ('IPVPN','MISN')"
'''

# Function for prediction and validation

def prediction(df, j = ""):

    # Label encoding
    
    df1 = df.iloc[:, 1:]
    for i in df1.columns:
        if (df1[i].dtype == 'object'):
            let = joblib.load(le_path + i + j + '.pkl')
            df1[i] = let.transform(df1[i])

    x = df1.drop('No_of_incidents', axis=1)

    # Loading models

    DTR = joblib.load(model_path+'1_DTR_'+ j +'.pkl')
    KNNR = joblib.load(model_path+'1_KNNR_'+ j +'.pkl')
    RFR = joblib.load(model_path+'1_RFR_'+ j +'.pkl')

    df['discover source'] = j

    # Prediction

    DTR_pred = np.array(DTR.predict(x))
    KNNR_pred = np.array(KNNR.predict(x))
    RFR_pred = np.array(RFR.predict(x))


    df['Predicted_final'] = ((2*DTR_pred) + KNNR_pred + RFR_pred)/4
    df['Predicted_final'] = df['Predicted_final'].astype('int64')
    
    # std = (df['No_of_incidents']-df['Predicted_final']).std()
    # print("\n\nStandard deviation of predicted incidents : ",std,"\n\n")

    return df

# loading dataset

# INC_df = pd.read_sql_query(query, con)
INC_df = pd.read_csv(r'A:\MLOps\Incident prediction\IncidentPredictionMLOps_v1\input\INC_df_2.csv')

INC_df.columns = ['discovery','discovery source','customerseverity','devicetype_en','service name_en','province']

# Preparing data-set for prediction

INC_df = data_prep(INC_df)


# Creating  datelist for validation

current_dt = datetime.datetime.strptime(date, "%d-%m-%Y")
date_list = [current_dt - datetime.timedelta(days=x) for x in range(1,1+int(val_range))]
date_list = [x.date() for x in date_list]

# Fetching matched date rows from df.

INC_df_2  = INC_df[INC_df['date'].isin(date_list)]

# Removing outliers and making prediction and validation

validation_df = INC_df_2[(INC_df_2['No_of_incidents'] < 50)].drop(['discovery source'],axis=1)

print('Predicting data for validation...')

output = prediction(validation_df, j = 'User Created')
output = output[['date', 'month', 'day', 'BH', 'province', 'customerseverity', 'devicetype_en',
                 'service name_en','discover source', 'No_of_incidents', 'Predicted_final']]

# Exporting output to DB
'''
params = urllib.parse.quote_plus(r'Driver={SQL Server};'
                                'Server='+server+';'
                                'Database='+DB+';'
                                'uid='+ID+';pwd='+PWD)
                                
conn_str = 'mssql+pyodbc:///?odbc_connect={}'.format(params)

engine = create_engine(conn_str)
output.to_sql(name='IncidentValidation', con=engine, if_exists='replace', index=False)
'''
output.to_csv(r'A:\MLOps\Incident prediction\IncidentPredictionMLOps_v1\Output\validation_output.csv')
print('Validation done.')