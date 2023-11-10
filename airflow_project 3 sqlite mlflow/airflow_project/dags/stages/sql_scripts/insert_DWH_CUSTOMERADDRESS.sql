INSERT OR IGNORE INTO DWH_CustomerAddresses (
customer_id,
address,
postcode, 
state,
country,
property_valuation)

SELECT cast(customer_id as bigint) as customer_id,
address,postcode,state,country,
cast(property_valuation as INTEGER) as property_valuation
FROM   STG_CUSTOMERADDRESS;
