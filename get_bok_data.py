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
    

with open('data/bok_api_key.txt', 'r') as f:
    bok_key = f.read()

code_names = ['101Y004', '101Y015', '101Y008', '111Y004', '111Y006', 
              '112Y002', '104Y014', '111Y008', '104Y016', '111Y009', 
              '141Y005', '151Y001', '151Y004', '131Y016', '104Y017',
              '722Y001', '721Y001', '121Y002', '121Y006', '121Y004',
              '121Y007', '901Y014', '901Y055', '901Y015', '191Y001',
              '901Y013', '200Y001', '200Y002', '301Y013', '301Y017',
              '731Y004', '731Y005', '901Y011', '901Y012', '303Y002',
              '303Y005', '403Y001', '403Y003', '403Y005', '901Y060',
              '732Y001', '402Y014', '401Y015', '901Y062', '511Y002',
              '511Y003', '513Y001', '521Y001', '901Y066', '901Y027',
              '901Y052', '902Y006', '902Y007', '902Y008', '902Y020',
              '902Y021', '902Y002']
periods_list = ['M', 'M', 'M', 'M', 'M', 
                'M', 'M', 'M', 'M', 'M', 
                'M', 'Q', 'Q', 'Q', 'M',
                'M', 'M', 'M', 'M', 'M',
                'M', 'M', 'M', 'M', 'M',
                'M', 'A', 'Q', 'M', 'M',
                'M', 'M', 'M', 'M', 'M',
                'M', 'M', 'M', 'M', 'M',
                'M', 'M', 'M', 'M', 'M',
                'M', 'M', 'M', 'M', 'M',
                'M', 'M', 'M', 'M', 'M',
                'M', 'M']

### 호출하려는 OpenAPI URL를 정의합니다.
#url_get_code_name = f'https://ecos.bok.or.kr/api/StatisticTableList/{bok_key}/json/en/1/20000'
##url = f'https://ecos.bok.or.kr/api/KeyStatisticList/{bok_key}/json/en/1/100'
#response = requests.get(url_get_code_name)
### http 요청이 성공했을때 API의 리턴값을 가져옵니다.
#if response.status_code != 200: 
#    print('Error code:  ', response.status_code)
#data_rows = response.json()['StatisticTableList']['row']
#for row in data_rows:
#    stat_code = row['STAT_CODE']
#    if stat_code not in code_names and row['CYCLE'] is not None:
#        code_names.append(row['STAT_CODE'])
#        periods_list.append(row['CYCLE'])

names_for_check = ['STAT_NAME', 'ITEM_NAME1', 'ITEM_NAME2', 'ITEM_NAME3', 'ITEM_NAME4']
code_names_for_check = ['STAT_CODE', 'ITEM_CODE1', 'ITEM_CODE2', 'ITEM_CODE3', 'ITEM_CODE4']
saved_code_names = []
for code, period in zip(code_names, periods_list):
    if period == 'M':
        start = '200001'
        end = '202210'
    elif period == 'Q':
        start = '2000Q1'
        end = '2022Q3'
    elif period == 'A':
        start = '2000'
        end = '2022'
    elif period == 'S':
        start = '2000S1'
        end = '2022S2'
    elif period == 'D':
        start = '20000101'
        end = '20221101'
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
        if full_code_name in saved_code_names:
            df = pd.read_csv(f'data/bok/economy_data_{full_code_name}.csv', index_col=0)
            columns = list(df.columns)
        else:
            saved_code_names.append(full_code_name)
            columns = ['data_name', 'date', 'value']
            df = pd.DataFrame(columns=columns)
        row = pd.DataFrame(np.zeros([1, len(columns)]), columns=columns)
        row['data_name'] = data_name
        row['date'] = get_date(data_time, period)
        row['value'] = data_value
        df = pd.concat([df, row], ignore_index=True)
        df.to_csv(f'data/bok/economy_data_{full_code_name}.csv')