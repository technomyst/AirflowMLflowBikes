DROP TABLE if exists STG_TRANSACTIONS;
CREATE TABLE if not exists STG_TRANSACTIONS(
transaction_id varchar(128), 
product_id varchar(128), 
customer_id varchar(128), 
transaction_date varchar(128), 
online_order varchar(128),
order_status varchar(128), 
brand varchar(128), 
product_line varchar(128), 
product_class varchar(128), 
product_size varchar(128), 
list_price varchar(128), 
standard_cost varchar(128), 
product_first_sold_date varchar(128));