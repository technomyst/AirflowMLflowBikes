CREATE TABLE if not exists DWH_Products(
id BIGINT primary key, 
brand varchar(128),
class varchar(128),
line varchar(128),
"size" varchar(128),
standard_cost real,
list_price real,
product_first_sold_date INTEGER);