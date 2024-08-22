#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 00:46:31 2024

@author: noob
"""
from bs4 import BeautifulSoup
import requests

url = 'https://eg.hatla2ee.com/ar/car/'
response = requests.get(url)


def get_page_list(response):
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        divs = soup.find_all('div', class_='newCarListUnit_header')
        collected_items = []
        
        for div in divs:
            span = div.find('span')
            if span:
                a = span.find('a')
                
                if a:
                    href = a.get('href')
                    collected_items.append({'href': href})

        return collected_items
    else:
        print(f"Failed to retrieve page. Status code: {response.status_code}")
        return None


links_list = get_page_list(response)


url = 'https://eg.hatla2ee.com/ar/car/page/'
counter = 2
while counter <= 1172:
    print("PAGE : "+ str(counter))
    response = requests.get(url+str(counter))
    links_list.append(get_page_list(response))
    counter += 1 


import json
# with open('links.txt', 'w', encoding='utf-8') as file:
#     for link_page in links_list:
#         file.write(json.dumps(link_page) + "\n")
        
data = []
with open("links.txt", "r") as file:
    for line in file:
        data.append(json.loads(line.strip()))



url = 'https://eg.hatla2ee.com'
car_list = []
counter = 35819
count = 1
flag = True
for page_link in data:
    for car_link in page_link:
        if(flag):
            if count != counter:
                count +=1
                continue
            else:
                flag = False
        try:
            print("car : " + str(counter))
            response = requests.get(url+car_link['href'])
            soup = BeautifulSoup(response.content, 'html.parser')
            price_span = soup.find_all('span', class_='usedUnitCarPrice')[0]
            price = price_span.get_text()
    
            details_div = soup.find_all('div', class_='DescDataRow')[0]
            details_div_spans =  soup.find_all('span', class_='DescDataVal')
            brand = details_div_spans[0].get_text()
            model = details_div_spans[1].get_text()
    
            if len(details_div_spans) == 10:
                year = details_div_spans[3].get_text()
                km = details_div_spans[4].get_text()
                fuel_type = details_div_spans[5].get_text()
            else:
                year = details_div_spans[2].get_text()
                km = details_div_spans[3].get_text()
                fuel_type = details_div_spans[4].get_text()
            car_list.append({'brand': brand ,'model' : model , 'year':year , 'km' : km , 'fuel_type' : fuel_type ,'price' : price})
        except:
            print("skip " + str(counter))
        counter += 1

        
with open('cars_all.txt', 'w', encoding='utf-8') as file:
    for car in car_list:
        file.write(json.dumps(car) + "\n")