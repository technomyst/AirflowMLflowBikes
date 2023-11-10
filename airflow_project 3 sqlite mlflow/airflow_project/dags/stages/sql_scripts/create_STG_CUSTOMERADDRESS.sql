DROP TABLE if exists STG_CUSTOMERADDRESS;
CREATE TABLE if not exists STG_CUSTOMERADDRESS(
customer_id varchar(128),
address varchar(500),
postcode varchar(128),
state varchar(255),
country varchar(255),
property_valuation varchar(128));


