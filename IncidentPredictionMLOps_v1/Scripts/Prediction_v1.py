
# Loading the libraries
 

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
# import pyodbc
import numpy as np
# from sklearn.externals import joblib
# import sklearn.external.joblib as extjoblib
import configparser
# from sqlalchemy import create_engine
import calendar
import datetime
import urllib
import random
import joblib
# from datetime import datetime

# Reading configuration file

config = configparser.RawConfigParser()

try:
    config.read('Configuration.txt')
except Exception as e:
    print(str(e))
try:
    # server = config.get('SQL_server', 'server')
    # DB = config.get('SQL_server', 'DB')
    # ID = config.get('SQL_server', 'ID')
    # PWD = config.get('SQL_server', 'PWD')
    # query = config.get('SQL_server', 'sql_query')

    model_path = config.get('Paths', 'model_path')
    metric_path = config.get('Paths', 'metric_path')
    le_path = config.get('Paths', 'le_path')
    input_path = config.get('Paths', 'input_path')
    output = config.get('Paths', 'output_path')
    date = config.get('Date', 'prediction_date')
    pre_range = config.get('Date', 'prediction_range')

except Exception as e:
    print('Could not read configuration file. {}'.format(str(e)))

# conecting to database server retriving data

print("Connecting to data")

'''
con = pyodbc.connect(r'Driver={SQL Server};'
                                'Server='+server+';'
                                'Database='+DB+';'
                                'uid='+ID+';pwd='+PWD)
'''

# INC_df = pd.read_sql_query(query, con)
INC_df = pd.read_csv(r'../Input/INC_df_2.csv')

# Function for prediction


def prediction(df, j='User Created'):

    # Label encoding
    
    df1 = df.iloc[:, 1:]
    for i in df1.columns:
        if (df1[i].dtype == 'object'):
            let = joblib.load(le_path + i + j + '.pkl')
            df1[i] = let.transform(df1[i])

    x = df1

    # Loading models

    DTR = joblib.load(model_path+'1_DTR_'+ j +'.pkl')
    KNNR = joblib.load(model_path+'1_KNNR_'+ j +'.pkl')
    RFR = joblib.load(model_path+'1_RFR_'+ j +'.pkl')

    df['discover source'] = j

    # Predicting data

    DTR_pred = np.array(DTR.predict(x))
    KNNR_pred = np.array(KNNR.predict(x))
    RFR_pred = np.array(RFR.predict(x))


    df['Predicted_incidents'] = ((2*DTR_pred) + KNNR_pred + RFR_pred)/4
    df['Predicted_incidents'] = df['Predicted_incidents'].astype('int64')

    return df

# Creating date list for prediction

current_dt = datetime.datetime.strptime(date, "%d-%m-%Y")
date_list = [current_dt + datetime.timedelta(days=x) for x in range(1,1+int(pre_range))]
date_list = [x.date() for x in date_list]


# Fetching 5 combination randomly from past incidents

date_list_2 = [current_dt - datetime.timedelta(days=x) for x in range(1,101)]
str_dates = [date_obj.strftime('%Y-%m-%d') for date_obj in date_list_2]

custom_date = list(INC_df[INC_df['date'].isin(str_dates)]['date'])
random_5_dt = random.choices(custom_date,k=5)

# Creating prediction data

INC_df = INC_df[INC_df['date'].isin(random_5_dt)]

# removing outliers

df_1 = INC_df[(INC_df['No_of_incidents'] < 30)]

# Updating date, month and day on picked combinations

df_3 = pd.DataFrame()
for i,j in zip(date_list,random_5_dt):
    df_2 = df_1[df_1['date']==j]
    df_2['date'] = i
    df_2['month'] = calendar.month_abbr[i.month]
    df_2['day'] = calendar.day_name[i.weekday()]
    df_3 = pd.concat([df_3,df_2])


# df_1 = INC_df[(INC_df['No_of_incidents'] < 50)].drop(['discovery source','date','month','day','No_of_incidents'],axis=1)

# df_1.drop_duplicates(keep='first',inplace=True)
# df_n3 = pd.DataFrame()
# for i in date_list:
    # df_2 = df_1
    # df_2['date'] = i
    # df_2['month'] = calendar.month_abbr[i.month]
    # df_2['day'] = calendar.day_name[i.weekday()]
    # df_n3 = pd.concat([df_n3,df_2])
    

prediction_df = df_3[['date', 'month', 'day', 'BH', 'province', 'customerseverity', 'devicetype_en','service name_en']]

# Predicting incidents

print('Predicting data...')

output = prediction(prediction_df)

# Exporting output to DB

output = output[['date', 'month', 'day', 'BH', 'province', 'customerseverity', 'devicetype_en',
                 'service name_en','discover source', 'Predicted_incidents']]

'''
params = urllib.parse.quote_plus(r'Driver={SQL Server};'
                                'Server='+server+';'
                                'Database='+DB+';'
                                'uid='+ID+';pwd='+PWD)
                                
conn_str = 'mssql+pyodbc:///?odbc_connect={}'.format(params)

engine = create_engine(conn_str)
output.to_sql(name='IncidentPredicition_N5', con=engine, if_exists='replace', index=False)
'''

output.to_csv(r'../Output/prediction_output.csv')

print('Prediction done.')

