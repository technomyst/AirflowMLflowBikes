import pandas as pd
import os
import shutil
from stages.etl.etl_task import EtlTask
import logging
logging.basicConfig(level=logging.DEBUG, filename='myapp.log', format='%(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class Extract_data:

    def __init__(self, root_path, connection ,root_path_input_data,root_path_archive,tables_info,db_path,path_sql_scripts):
        self.root_path = root_path
        self.connection = connection

        self.root_path_input_data = root_path_input_data
        self.root_path_archive = root_path_archive
        self.tables_info = tables_info
        self.db_path = db_path
        self.path_sql_scripts=path_sql_scripts

        self.success_stage = False
        self.success_dwh = False


    def extract_data(self):
        buf = []
        files = os.listdir(self.root_path)
        for i in files:
            buf.append(pd.read_csv(f'{self.root_path}/{i}'))
        if len(buf) == 0:
            return pd.DataFrame()
        return pd.concat(buf)

    def transform(self, data, subset='Name'):
        data = data.dropna(subset=subset)
        return data

    def load(self, data, table_name):
        data.to_sql(table_name, con=self.connection, if_exists='replace')

    def create_tabel(self, table_name):
        data = self.extract_data()
        self.my_extract_data()
        '''test1=self.my_extract_data()'''
        data = self.transform(data)
        self.load(data, table_name)
        return table_name

    def my_extract_data(self):
        input_path = self.root_path_input_data
        archive_path = self.root_path_archive

        input_files = os.listdir(input_path)
        archive_files = os.listdir(archive_path)

        new_files = [x for x in input_files if x + '.dump' not in archive_files]

        for i in new_files:
            logger.info(input_path)
            logger.info(i)
            self.stg_and_dwh(input_path, i)
            shutil.copy(f"{input_path}{i}", f"{archive_path}{i}.dump")

    def my_extract_data_to_stg(self):

        input_path = self.root_path_input_data
        archive_path = self.root_path_archive
        self.success_stage = False
        input_files = os.listdir(input_path)
        archive_files = os.listdir(archive_path)

        task = EtlTask(self.db_path)

        new_files = [x for x in input_files if x + '.dump' not in archive_files]

        for i in new_files:
            logger.info(input_path)
            logger.info(i)
            data = self.read_data(f'{input_path}{i}')
            file_name = i
            table_stg_list = []
            dwh_create_script_list = []
            dwh_insert_script_list = []
            if 'CustomerDemographic' in file_name:
                table_stg_list.append('STG_CUSTOMERDEMOGRAPHIC')
                dwh_create_script_list.append(self.tables_info['Customers']['dwh_create'])
                dwh_insert_script_list.append(self.tables_info['Customers']['dwh_insert'])
            elif 'CustomerAddress' in file_name:
                table_stg_list.append('STG_CUSTOMERADDRESS')
                dwh_create_script_list.append(self.tables_info['CustomerAddresses']['dwh_create'])
                dwh_insert_script_list.append(self.tables_info['CustomerAddresses']['dwh_insert'])
            elif 'Transactions' in file_name:
                table_stg_list.append('STG_TRANSACTIONS')
                dwh_create_script_list.append(self.tables_info['Transactions']['dwh_create'])
                dwh_insert_script_list.append(self.tables_info['Transactions']['dwh_insert'])

                dwh_create_script_list.append(self.tables_info['Products']['dwh_create'])
                dwh_insert_script_list.append(self.tables_info['Products']['dwh_insert'])
            elif 'NewCustomerList' in file_name:
                table_stg_list.append('STG_NEWCUSTOMERLIST')
                dwh_create_script_list = []
                dwh_insert_script_list = []

            for j in table_stg_list:
                task.initialize_stg(j,self.path_sql_scripts)
                task.insert_data_to_stg(j, data)

        self.success_stage = True

        '''self.stg_and_dwh(input_path, i)'''
        '''shutil.copy(f"{input_path}{i}", f"{archive_path}{i}.dump")'''

        return self.success_stage


    def my_extract_filesdata_from_stg_to_dwh(self):
        input_path = self.root_path_input_data
        archive_path = self.root_path_archive
        self.success_dwh = False

        input_files = os.listdir(input_path)
        archive_files = os.listdir(archive_path)

        task = EtlTask(self.db_path)

        new_files = [x for x in input_files if x + '.dump' not in archive_files]

        for i in new_files:
            logger.info(input_path)
            logger.info(i)
            data = self.read_data(f'{input_path}{i}')
            file_name = i
            table_stg_list = []
            dwh_create_script_list = []
            dwh_insert_script_list = []
            if 'CustomerDemographic' in file_name:
                table_stg_list.append('STG_CUSTOMERDEMOGRAPHIC')
                dwh_create_script_list.append(self.tables_info['Customers']['dwh_create'])
                dwh_insert_script_list.append(self.tables_info['Customers']['dwh_insert'])
            elif 'CustomerAddress' in file_name:
                table_stg_list.append('STG_CUSTOMERADDRESS')
                dwh_create_script_list.append(self.tables_info['CustomerAddresses']['dwh_create'])
                dwh_insert_script_list.append(self.tables_info['CustomerAddresses']['dwh_insert'])
            elif 'Transactions' in file_name:
                table_stg_list.append('STG_TRANSACTIONS')
                dwh_create_script_list.append(self.tables_info['Transactions']['dwh_create'])
                dwh_insert_script_list.append(self.tables_info['Transactions']['dwh_insert'])

                dwh_create_script_list.append(self.tables_info['Products']['dwh_create'])
                dwh_insert_script_list.append(self.tables_info['Products']['dwh_insert'])
            elif 'NewCustomerList' in file_name:
                table_stg_list.append('STG_NEWCUSTOMERLIST')
                dwh_create_script_list = []
                dwh_insert_script_list = []

            for j in dwh_create_script_list:
                task.make_dwh(j)
                logger.info(j)
            for j in dwh_insert_script_list:
                task.make_dwh(j)
                logger.info(j)

        self.success_dwh = True
        self.send_files_to_archive()
        return self.success_dwh

    def send_files_to_archive(self):

        input_path = self.root_path_input_data
        archive_path = self.root_path_archive

        input_files = os.listdir(input_path)
        archive_files = os.listdir(archive_path)

        new_files = [x for x in input_files if x + '.dump' not in archive_files]

        logger.info("Отправка файлов в архив")
        for i in new_files:
            logger.info(input_path)
            logger.info(i)
            '''to do'''
            if self.success_dwh:
                shutil.copy(f"{input_path}{i}", f"{archive_path}{i}.dump")
            else:
                '''to do'''
                pass


    def my_extract_files_to_stg_dwh(self,folder_path, file_name):
        tables_info = self.tables_info
        data = self.read_data(f'{folder_path}{file_name}')

        task = EtlTask(self.db_path)

        table_stg_list = []
        dwh_create_script_list = []
        dwh_insert_script_list = []
        if 'CustomerDemographic' in file_name:
            table_stg_list.append('STG_CUSTOMERDEMOGRAPHIC')
            dwh_create_script_list.append(tables_info['Customers']['dwh_create'])
            dwh_insert_script_list.append(tables_info['Customers']['dwh_insert'])
        elif 'CustomerAddress' in file_name:
            table_stg_list.append('STG_CUSTOMERADDRESS')
            dwh_create_script_list.append(tables_info['CustomerAddresses']['dwh_create'])
            dwh_insert_script_list.append(tables_info['CustomerAddresses']['dwh_insert'])
        elif 'Transactions' in file_name:
            table_stg_list.append('STG_TRANSACTIONS')
            dwh_create_script_list.append(tables_info['Transactions']['dwh_create'])
            dwh_insert_script_list.append(tables_info['Transactions']['dwh_insert'])

            dwh_create_script_list.append(tables_info['Products']['dwh_create'])
            dwh_insert_script_list.append(tables_info['Products']['dwh_insert'])
        elif 'NewCustomerList' in file_name:
            table_stg_list.append('STG_NEWCUSTOMERLIST')
            dwh_create_script_list = []
            dwh_insert_script_list = []

        for i in table_stg_list:
            task.initialize_stg(i)
            task.insert_data_to_stg(i, data)
        for i in dwh_create_script_list:
            task.make_dwh(i)
        for i in dwh_insert_script_list:
            task.make_dwh(i)



    def read_data(self,path: str) -> pd.DataFrame:
        name = path.split('.')
        if name[-1] == 'xlsx':
            return pd.read_excel(path)
        elif name[-1] == 'txt':
            return pd.read_csv(path, sep=';')
        elif name[-1] == 'dump':
            return read_data(".".join(name[:-1]))
