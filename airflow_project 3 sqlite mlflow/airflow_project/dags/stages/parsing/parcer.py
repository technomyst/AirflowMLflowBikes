import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import logging
logging.basicConfig(level=logging.DEBUG, filename='myapp.log', format='%(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class Parcer:

    def __init__(self,categories=[],filters=[]):
        self.website='https://velosport.ru/'
        self.url_before_category = 'https://velosport.ru/catalog/'
        self.categories=categories
        if not isinstance(self.categories, list):
            self.categories = [self.categories]
        if not len(self.categories):
            self.categories=['ekipirovka/noski_golfy_i_getry',
                        'ekipirovka/shlemy',
                        'aksecsuary/nasosy'
                        'aksecsuary/fonari',
                        'aksecsuary/flyagi',
                        'ekipirovka/ryukzaki_i_sumki']
        logger.info(self.categories)
        self.website_filters=filters
        if not isinstance(self.website_filters, list):
            self.website_filters = [self.website_filters]
        if not len(self.website_filters):
            self.website_filters = ['popular',
                               'new',
                               'discount']
        logger.info(self.website_filters)

    unwanted_chars = ['\n', '\t', '\xa0']

    def clean_up_text(self,text, unwanted_chars=unwanted_chars):
        for char in unwanted_chars:
            text = text.replace(char, '')
        return text

    def get_data(self,category,filter):
        response = requests.get(f'{self.url_before_category}{category}/?sort={filter}')
        logger.info(f'{self.url_before_category}{category}/?sort={filter}')
        soup = BeautifulSoup(response.text, features="html.parser")
        all_items = soup.find_all('div', {"class": 'product-item'})
        logger.info(len(all_items))
        if len(all_items):
            return all_items
        else:
            return None

    def create_dataframe(self, items):
        d = {'name': [],
             'type': [],
             'count': [],
             'count_measure': [],
             'source_id': [],
             'price_current': [],
             'price_current_currency': [],
             'price_old': [],
             'price_old_currency': [],
             'url_detail': [],
             'source': [],
             'load_date': []
             }

        for i in items:
            name = i.find_all('h3', {'class': 'product-item-title'})
            if len(name):
                name_a = name[0].find_all('a')
                if len(name_a):
                    name_a_title = name_a[0]['title']
                    logger.info(f'name_a_title={name_a_title}')
                    d['name'].append(name_a_title)
                    url_detail = name_a[0]['href']
                    d['url_detail'].append(url_detail)

            item_type = i.find_all('div', {'class': 'section-name'})
            if len(item_type):
                d['type'].append(item_type[0].text)

            item_count = i.find_all('span', {'class': 'product-item-quantity'})
            logger.info(f'item_count={item_count}')
            if len(item_count):
                logger.info(f'item_count[0].text={item_count[0].text}')
                if len(item_count[0].text.strip()):
                    count_cleaned = self.clean_up_text(item_count[0].text)
                    logger.info(f'count_cleaned={count_cleaned}')
                    count_cleaned_value, count_cleaned_measure = count_cleaned.split(' ')
                    d['count'].append(count_cleaned_value)
                    d['count_measure'].append(count_cleaned_measure)
                else:
                    d['count'].append("0")
                    d['count_measure'].append("-")
            else:
                d['count'].append("0")
                d['count_measure'].append("-")

            source_id = i.find_all('a', {'class': 'favorite-item-button'})
            if len(source_id):
                d['source_id'].append(source_id[0]['data-item'])

            price_cur = i.find_all('span', {'class': 'product-item-price-current'})
            if len(price_cur):
                if len(price_cur[0].text.strip()):
                    price_cur_cleaned = self.clean_up_text(price_cur[0].text)
                    price_cur_cleaned_value, price_cur_cleaned_currency = price_cur_cleaned.split()
                    d['price_current'].append(price_cur_cleaned_value)
                    d['price_current_currency'].append(price_cur_cleaned_currency)
                else:
                    d['price_current'].append("0")
                    d['price_current_currency'].append("-")
            else:
                d['price_current'].append("0")
                d['price_current_currency'].append("-")
            price_old = i.find_all('span', {'class': 'product-item-price-old'})
            if len(price_old):
                if len(price_old[0].text.strip()):
                    price_old_cleaned = self.clean_up_text(price_old[0].text)
                    price_old_cleaned_value, price_old_cleaned_currency = price_old_cleaned.split(' ')
                    d['price_old'].append(price_old_cleaned_value)
                    d['price_old_currency'].append(price_old_cleaned_currency)
                else:
                    d['price_old'].append("0")
                    d['price_old_currency'].append("-")
            else:
                d['price_old'].append("0")
                d['price_old_currency'].append("-")

            d['source'].append(self.website)
            d['load_date'].append(datetime.now())

            '''logger.info(d);'''
        return pd.DataFrame(d)

    def take_website_data(self):
        buf = []
        for i in self.categories:
            logger.info(f'i={i}')
            for j in self.website_filters:
                logger.info(f'j={j}')
                data = self.get_data(i,j)
                if data is None:
                    continue
                else:
                    df = self.create_dataframe(data)
                    df['category'] = i
                    df['webfilter'] = j
                    buf.append(df)
        full_data = pd.concat(buf)
        return full_data





