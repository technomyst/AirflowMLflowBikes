CREATE TABLE if not exists DWH_Customers(
    id                         BIGINT primary key,
    first_name                          varchar(128),
    last_name                           varchar(128),    
    gender                              varchar(128),
    date_of_birth                       date,
    past_3_years_bike_related_purchases INTEGER,
    job_title                           varchar(128),
    job_industry_category               date,
    wealth_segment                      varchar(128),
    deceased_indicator                  INTEGER,
    owns_car                            INTEGER);
