CREATE TABLE if not exists DWH_Transactions(
    id                         BIGINT primary key,
    product_id                 BIGINT,
    customer_id                BIGINT,    
    transaction_date           date,
    is_online_order            INTEGER,
    order_status               varchar(128),
FOREIGN KEY(product_id) REFERENCES DWH_Products(id),
FOREIGN KEY(customer_id) REFERENCES DWH_Customers(id));
