CREATE TABLE if not exists DWH_CustomerAddresses(
    customer_id                        BIGINT,
    address                             varchar(128),    
    postcode                            varchar(128),
    state                               varchar(128),
    country                             varchar(128),
    property_valuation                  INTEGER,
FOREIGN KEY(customer_id) REFERENCES DWH_Customers(id));