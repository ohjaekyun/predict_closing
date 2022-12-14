import requests
import xml.etree.ElementTree as ET 
import xml.dom.minidom
import json
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta


def get_date(data_time, period):
    year = int(data_time[:4])
    if period == 'M':
        month = int(data_time[4:6])
        return date(year, month, 1) + relativedelta(months=1) - timedelta(days=1)

    elif period == 'Q':
        quater = int(data_time[5])
        if quater == 1:
            return date(year, 3, 31)
        elif quater == 2:
            return date(year, 6, 30)
        elif quater == 3:
            return date(year, 9, 30)
        else:
            return date(year, 12, 31)

    elif period == 'A':
        return date(year, 12, 31)

    elif period == 'S':
        half = int(data_time[5])
        if half == 2:
            return date(year, 6, 30)
        else:
            return date(year, 12, 31)

    elif period == 'D':
        month = int(data_time[4:6])
        day = int(data_time[6:8])
        return date(year, month, day)

    else:
        return None
    

def get_year_and_month(data_time, period):
    year = int(data_time[:4])
    if period == 'M':
        month = int(data_time[4:6])
        return year, month

    elif period == 'Q':
        quater = int(data_time[5])
        if quater == 1:
            return year, 3
        elif quater == 2:
            return year, 6
        elif quater == 3:
            return year, 9
        else:
            return year, 12

    elif period == 'A':
        return year, 12

    elif period == 'S':
        half = int(data_time[5])
        if half == 2:
            return year, 6
        else:
            return year, 12

    elif period == 'D':
        month = int(data_time[4:6])
        day = int(data_time[6:8])
        return year, month

    else:
        return None


with open('data/bok_api_key.txt', 'r') as f:
    bok_key = f.read()

code_names = ['101Y004', '101Y015', '101Y008', '111Y004', '111Y006', 
              '112Y002', '104Y014', '111Y008', '104Y016', '111Y009', 
              '141Y005', '151Y002', '131Y016', '104Y017',
              '722Y001', '721Y001', '121Y002', '121Y006', '121Y004',
              '121Y007', '901Y014', '901Y055', '901Y015', '191Y001',
              '901Y013', '200Y002', '301Y013', '301Y017',
              '731Y004', '731Y005', '901Y011', '901Y012', '303Y002',
              '303Y005', '403Y001', '403Y003', '403Y005', '901Y060',
              '732Y001', '402Y014', '401Y015', '901Y062', '511Y002',
              '511Y003', '513Y001', '521Y001', '901Y066', '901Y027',
              '901Y052', '902Y006', '902Y007', '902Y008', '902Y020',
              '902Y021', '902Y002']
periods_list = ['M', 'M', 'M', 'M', 'M', 
                'M', 'M', 'M', 'M', 'M', 
                'M', 'M', 'Q', 'M',
                'M', 'M', 'M', 'M', 'M',
                'M', 'M', 'M', 'M', 'M',
                'M', 'Q', 'M', 'M',
                'M', 'M', 'M', 'M', 'M',
                'M', 'M', 'M', 'M', 'M',
                'M', 'M', 'M', 'M', 'M',
                'M', 'M', 'M', 'M', 'M',
                'M', 'M', 'M', 'M', 'M',
                'M', 'M']

names_for_check = ['STAT_NAME', 'ITEM_NAME1', 'ITEM_NAME2', 'ITEM_NAME3', 'ITEM_NAME4']
code_names_for_check = ['STAT_CODE', 'ITEM_CODE1', 'ITEM_CODE2', 'ITEM_CODE3', 'ITEM_CODE4']
saved_code_names = []
base_year = 2006
df_dict = {}
for code, period in zip(code_names, periods_list):
    if period == 'M':
        start = f'{base_year}01'
        end = '202209'
    elif period == 'Q':
        start = f'{base_year}Q1'
        end = '2022Q3'
    elif period == 'A':
        start = f'{base_year}'
        end = '2022'
    elif period == 'S':
        start = f'{base_year}S1'
        end = '2022S2'
    elif period == 'D':
        start = f'{base_year}0101'
        end = '20220930'
    else:
        continue
    url = f'https://ecos.bok.or.kr/api/StatisticSearch/{bok_key}/json/en/1/50000/{code}/{period}/{start}/{end}/'
    response = requests.get(url)
    if response.status_code != 200:
        continue
    res_json = response.json()
    if 'StatisticSearch' not in res_json:
        continue
    datas = res_json['StatisticSearch']['row']
    for data in datas:
        data_value = data['DATA_VALUE']
        if data_value is None or data_value == '':
            continue
        curr_names = []
        for name0 in names_for_check:
            data_for_name = data[name0]
            if data_for_name is not None:
                curr_names.append(data_for_name)
        data_name = '_'.join(curr_names)
        data_time = data['TIME']
        curr_codes = []
        for code0 in code_names_for_check:
            data_for_code = data[code0]
            if data_for_code is not None:
                curr_codes.append(data_for_code)
        full_code_name = '_'.join(curr_codes)
        if full_code_name in df_dict:
            df = df_dict[full_code_name]
            columns = list(df.columns)
        else:
            saved_code_names.append(full_code_name)
            columns = ['year', 'month', 'data_name', 'value']
            df = pd.DataFrame(columns=columns)
        row = pd.DataFrame(np.zeros([1, len(columns)]), columns=columns)
        row['data_name'] = data_name
        year, month = get_year_and_month(data_time, period)
        if period == 'Q':
            row['year'] = year
            row['month'] = month - 2
            row['value'] = data_value
            df = pd.concat([df, row], ignore_index=True)
            row['year'] = year
            row['month'] = month - 1
            row['value'] = data_value
            df = pd.concat([df, row], ignore_index=True)
        row['year'] = year
        row['month'] = month
        row['value'] = data_value
        df = pd.concat([df, row], ignore_index=True)
        df_dict[full_code_name] = df
year1 = base_year
month1 = 1
year2 = 2022
month2 = 6
month3 = 9
valid_dataframes = []
idx = 0
for code_name, df in df_dict.items():
    df0 = df[df['year'] == year1]
    df1 = df0[df0['month'] == month1]
    df00 = df[df['year'] == year2]
    df2 = df00[df00['month'] == month2]
    df3 = df00[df00['month'] == month3]
    if len(df1) > 0 and len(df3) > 0:
        data_name = df['data_name'].iloc[-1]
        data_name = data_name[data_name.find(' ') + 1:]
        df = df.drop(columns=['data_name'])
        df = df.rename(columns={'value': data_name}, inplace=False)
        if idx == 0:
            if len(df3) == 0:
                df7 = df00[df00['month'] == 7]
                if len(df7) == 0:
                    row1 = pd.DataFrame([year2, 7, np.nan], columns=df.columns)
                    df = pd.concat([df, row1], ignore_index=True)
                df8 = df00[df00['month'] == 8]
                if len(df8) == 0:
                    row1 = pd.DataFrame([year2, 8, np.nan], columns=df.columns)
                    df = pd.concat([df, row1], ignore_index=True)
                row1 = pd.DataFrame([year2, 9, np.nan], columns=df.columns)
                df = pd.concat([df, row1], ignore_index=True)
            idx = 1
        else:
            df = df.drop(columns=['year', 'month'])
            if len(df3) == 0:
                df7 = df00[df00['month'] == 7]
                if len(df7) == 0:
                    row1 = pd.DataFrame([np.nan], columns=df.columns)
                    df = pd.concat([df, row1], ignore_index=True)
                df8 = df00[df00['month'] == 8]
                if len(df8) == 0:
                    row1 = pd.DataFrame([np.nan], columns=df.columns)
                    df = pd.concat([df, row1], ignore_index=True)
                row1 = pd.DataFrame([np.nan], columns=df.columns)
                df = pd.concat([df, row1], ignore_index=True)
        valid_dataframes.append(df)
valid_df = pd.concat(valid_dataframes, axis=1)
valid_df.to_csv(f'data/total_economy_data.csv')
