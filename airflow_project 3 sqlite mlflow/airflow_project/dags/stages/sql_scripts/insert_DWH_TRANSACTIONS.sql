INSERT OR IGNORE INTO DWH_TRANSACTIONS (id,
                                   product_id,customer_id,
                                   transaction_date, is_online_order, 
                                             order_status)
SELECT cast(transaction_id as bigint) as transaction_id,
       cast(product_id as bigint) as product_id,
       cast(customer_id as bigint) as customer_id,
        date(transaction_date) as transaction_date,
        cast(online_order as INTEGER) as is_online_order,
        order_status
FROM   STG_TRANSACTIONS;


