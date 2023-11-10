INSERT OR IGNORE INTO DWH_Customers (id,
first_name,
last_name,
gender, 
date_of_birth, 
past_3_years_bike_related_purchases, 
job_title,
job_industry_category,
wealth_segment,
deceased_indicator,
owns_car)

SELECT cast(customer_id as bigint) as customer_id,
first_name,last_name,gender,
date(DOB) as date_of_birth,
cast(past_3_years_bike_related_purchases as INTEGER) as past_3_years_bike_related_purchases,
job_title,job_industry_category,wealth_segment,
CASE WHEN LOWER(deceased_indicator) = 'n' THEN 0 ELSE 1 END AS deceased_indicator,
CASE WHEN LOWER(owns_car) = 'yes' THEN 1 ELSE 0 END AS owns_car
FROM   STG_CUSTOMERDEMOGRAPHIC;

