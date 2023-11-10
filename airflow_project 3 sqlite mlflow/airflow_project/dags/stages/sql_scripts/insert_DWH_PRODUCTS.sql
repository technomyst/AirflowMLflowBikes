INSERT OR IGNORE INTO DWH_Products (
id,
brand,
class,
line,
size,
standard_cost,
list_price,
product_first_sold_date)

SELECT cast(product_id as bigint) as product_id,
       brand,
       product_class,
       product_line,
       product_size,
       cast(standard_cost as real) as standard_cost,
       cast(list_price as real) as list_price,
       cast(product_first_sold_date as INTEGER) as product_first_sold_date
FROM   STG_TRANSACTIONS;