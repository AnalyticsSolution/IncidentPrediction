

# Load the libraries


import pandas as pd
import calendar

def data_prep(df):

    df = df.fillna("NA")
    
    df["province"] = df["province"].str.upper().str.title()
    df["devicetype_en"] = df["devicetype_en"].str.upper().str.title()


    df_INC = df.copy()

    # Deriving new variable
    
    df_INC['discovery'] = pd.to_datetime(df_INC['discovery'], unit='s')
    df_INC['day'] = df_INC['discovery'].dt.day_name()
    df_INC['month'] = df_INC['discovery'].apply(lambda x: calendar.month_abbr[x.month])
    df_INC['BH'] = df_INC['discovery'].dt.time
    df_INC['date'] = df_INC['discovery'].dt.date

    # Binning device type_en
    
    list_1 = ['Router', 'Switch', 'Others']

    def binning_1(x):
        if x not in list_1:
            x = "Other 2"
        return x

    # Binning province
    
    df_INC['devicetype_en'] = df_INC['devicetype_en'].apply(binning_1)

    list_2 = ['Northwest Territories', 'Prince Edward Island', 'Yukon Territory', 'Nunavut']

    list_3 = ['New York', 'Florida', 'Virginia', 'California', 'Texas', 'Illinois', 'Pennsylvania',
            'Indiana', 'Massachusetts', 'Ohio', 'Wisconsin', 'Washington', 'New Jersey', 'North Carolina',
            'Colorado', 'Georgia', 'Nebraska', 'New Hampshire', 'Michigan', 'District of Columbia', 'Maryland',
            'Tennessee', 'Arizona', 'Delaware', 'Missouri', 'Oklahoma', 'South Carolina', 'Minnesota',
            'Arkansas', 'Hawaii', 'Connecticut', 'Alabama', 'Kansas', 'Maine', 'Rhode Island', 'Kentucky', 'Nevada',
            'Oregon', 'Alaska', 'Utah', 'South Dakota', 'Louisiana', 'Iowa', 'Idaho', 'West Virginia', 'Mississippi',
            'Wyoming','Na','State/Province not required','State/Province Not Required']

    def binning_2(x):
        if x in list_2:
            x = 'OtherCanada'
        if x in list_3:
            x = 'OtherUSA'
        if (x not in list_2) & (x not in list_3):
            x = 'OtherUSA'
        return x

    df_INC['province'] = df_INC['province'].apply(binning_2)

    df_INC['province'].fillna("OtherUSA", inplace=True)

    # Generating business hour and non BH variable
    
    def bh(x):
        y = str(x)
        y = y.split(':')
        y = float(".".join(y[:2]))
        z = 'NBH'
        if ((y > 8.30) & (y < 16.59)):
            z = 'BH'
        return z

    df_INC['BH'] = df_INC['BH'].apply(bh)

    # creating No of incidents in each group 
    
    df_final = pd.DataFrame(
        {'No_of_incidents': df_INC.groupby(['date', 'month', 'day', 'BH', 'province', 'discovery source',
                                              'customerseverity', 'devicetype_en',
                                              'service name_en']).size()}).reset_index()
    return df_final
    
