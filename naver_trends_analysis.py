import json
import pandas as pd
import os
import sys
import urllib.request
import ssl
import time
import requests
import hmac
import hashlib
import base64
from datetime import date


with open('text_keywords.txt', 'r') as f:
    keywords = f.read().split('\n')

with open('data/secrets.json') as f:
    dict_api = json.load(f)
dict_naver = dict_api['naver']
dict_naver_ad = dict_api['naver_ad']
client_id = dict_naver['client_id']
client_secret = dict_naver['secret']
ad_customer_id = dict_naver_ad['CUSTOMER_ID']
ad_access_license = dict_naver_ad['ACCESS_LICENSE']
ad_secret_key = dict_naver_ad['SECRET_KEY']

url = 'https://openapi.naver.com/v1/datalab/search'

def get_naver_trend_analysis(start_date, end_date, keywords):
    num_keywords = len(keywords)
    #keywords_string = '","'.join(keywords)
    #keywords_string = ''.join(['["', keywords_string, '"]'])
    body = f'{{"startDate":"{start_date}","endDate":"{end_date}","timeUnit":"month",'
    keyword_bodies = [body, '"keywordGroups":[']
    for keyword in keywords:
        keyword_body = f'{{"groupName":"{keyword}","keywords":["{keyword}"]}}' 
        keyword_bodies.append(keyword_body)
        keyword_bodies.append(',')
    keyword_bodies.pop()
    keyword_bodies.append(']}')
    body = ''.join(keyword_bodies)
    print(body)
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id", client_id)
    request.add_header("X-Naver-Client-Secret", client_secret)
    request.add_header("Content-Type", "application/json")
    context = ssl._create_unverified_context()
    response = urllib.request.urlopen(request, data=body.encode("utf-8"), context=context)
    rescode = response.getcode()
    if (rescode != 200):
        print('Error Code: ', rescode)
        return None
    response_body = response.read().decode('utf-8')
    return json.loads(response_body)['results']


def get_naver_ad_searches(keyword):
    BASE_URL = 'https://api.naver.com'
    timestamp = round(time.time() * 1000)
    uri = '/keywordstool'
    method = 'GET'
    signature = hmac.new(
        ad_secret_key.encode('utf-8'),
        msg = f'{timestamp}.{method}.{uri}'.encode('utf-8'), 
        digestmod = hashlib.sha256
        ).digest()

    headers = {
        "X-Timestamp": str(timestamp),
        "X-API-KEY": ad_access_license,
        "X-Customer": ad_customer_id,
        "X-Signature": base64.b64encode(signature).decode(),
        "Content-Type": "application/json"
    }
    response = requests.get(
        BASE_URL + uri + f'?hintKeywords={keyword}&showDetail=1',
        headers=headers
        )
    #request.add_header("X-API-KEY", ad_access_license)
    #request.add_header("X-Customer", ad_customer_id)
    #request.add_header("X-Signature", ad_secret_key)
    #request.add_header("X-Timestamp", timestamp)
    #request.add_header("Content-Type", "application/json")
    rescode = response.status_code
    if (rescode != 200):
        print('Error Code: ', rescode)
        return None
    response_body = response.json()
    return list(response_body.values())[0]


results = get_naver_trend_analysis('2014-01-01', str(date.today()), 'eight_percent', keywords)

dict_ad_searches = get_naver_ad_searches(keywords[0])
searches_count = dict_ad_searches[0]['monthlyMobileQcCnt'] + dict_ad_searches[0]['monthlyPcQcCnt']
data_df = pd.DataFrame.from_dict(results['data'])
data_df['keywords'] = ",".join(keywords)
recent_search_ratio = list(data_df['ratio'])[-1]
data_df['counts'] = searches_count / recent_search_ratio * data_df['ratio']
data_df.to_csv('eight_percent_trends.csv')