db = 'dags/mydb.db'
root_path = 'dags/archive'
root_path_input_data = 'dags/input_data/'
root_path_archive = 'dags/archive/'
path_sql_scripts = 'dags/stages/sql_scripts/'
output_data='dags/output_data/'

tables_info = {
    'Transactions': {'std': 'dags/stages/sql_scripts/create_STG_TRANSACTIONS.sql',
                    'dwh_create': 'dags/stages/sql_scripts/create_DWH_TRANSACTIONS.sql',
                    'dwh_insert': 'dags/stages/sql_scripts/insert_DWH_TRANSACTIONS.sql'},

    'CustomerAddresses': {'std': 'dags/stages/sql_scripts/create_STG_CUSTOMERADDRESS.sql',
                    'dwh_create': 'dags/stages/sql_scripts/create_DWH_CUSTOMERADDRESS.sql',
                    'dwh_insert': 'dags/stages/sql_scripts/insert_DWH_CUSTOMERADDRESS.sql'},

    'Customers': {'std': 'dags/stages/sql_scripts/create_STG_CUSTOMERDEMOGRAPHIC.sql',
                    'dwh_create': 'dags/stages/sql_scripts/create_DWH_CUSTOMERS.sql',
                    'dwh_insert': 'dags/stages/sql_scripts/insert_DWH_CUSTOMERS.sql'},

    'Products': {'dwh_create': 'dags/stages/sql_scripts/create_DWH_PRODUCTS.sql',
                'dwh_insert': 'dags/stages/sql_scripts/insert_DWH_PRODUCTS.sql'},

    'NewCustomerList': {'std': 'dags/stages/sql_scripts/create_STG_NEWCUSTOMERLIST.sql'},


}

select_dwh_for_model_script='''select * from DWH_Transactions t
join DWH_Customers c on c.id=t.customer_id
join DWH_CustomerAddresses ca on ca.customer_id=c.id
join DWH_Products p on p.id=t.product_id'''

