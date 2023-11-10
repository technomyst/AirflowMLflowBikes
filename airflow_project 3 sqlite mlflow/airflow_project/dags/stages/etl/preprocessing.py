import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from datetime import datetime,date


class Preprocessing:

    def __init__(self, categorical_feautures):
        self.categorical_feautures = categorical_feautures


    def process_data(self, data):
        data = data.loc[:, ~data.columns.duplicated()].copy()
        data = data.fillna(-1)
        today_date = datetime.now().date()

        data['customer_age_current'] = [
            (np.datetime64(today_date) - np.datetime64(x)).astype("<m8[Y]").astype(int) if x != -1 else -1 for x in
            data["date_of_birth"]]
        data['customer_age_on_transation_date'] = np.where(((data['date_of_birth'] != -1) & (data['transaction_date'] != -1)),
                                                         ((pd.to_datetime(data["transaction_date"],
                                                                          errors='coerce') - pd.to_datetime(
                                                             data["date_of_birth"], errors='coerce')).values.astype(
                                                             "<m8[Y]").astype(int)), -1)
        for i in self.categorical_feautures:
            lb = LabelEncoder()
            encode_data = lb.fit_transform(data[i].astype(str))
            data[f'{i}_lb'] = encode_data
        #named_lb = [f'{x}_lb' for x in categorical_feautures]
        data['order_status_int'] = data['order_status'].map(lambda x: 1 if x == 'Approved' else 0)

        data["transaction_year"] = pd.to_datetime(data["transaction_date"]).dt.year
        data["transaction_month"] = pd.to_datetime(data["transaction_date"]).dt.month
        data["transaction_week_of_year"] =pd.to_datetime(data["transaction_date"]).dt.strftime('%W').astype(int)

        data["postcode"] = data["postcode"].astype('Int64')

        needed_columns=['product_id'
,'customer_id'
,'is_online_order'
,'past_3_years_bike_related_purchases'
,'deceased_indicator'
,'owns_car'
,'postcode'
,'property_valuation'
,'standard_cost'
,'list_price'
,'product_first_sold_date'
,'order_status_int'
,'gender_lb'
,'job_title_lb'
,'job_industry_category_lb'
,'wealth_segment_lb'
,'state_lb'
,'brand_lb'
,'class_lb'
,'line_lb'
,'size_lb'
,'address_lb'
,'country_lb'
,'customer_age_current'
,'customer_age_on_transation_date'
,'transaction_year'
,'transaction_month'
,'transaction_week_of_year']
        data = data[needed_columns]
        return data

