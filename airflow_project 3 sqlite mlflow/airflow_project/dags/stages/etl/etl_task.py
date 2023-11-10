import pandas as pd
from stages.etl.connection import Connection

class EtlTask:

    def __init__(self, db_path):
        self.db_path = db_path

    def initialize_stg(self, stg_table_name: str,path_sql_scripts) -> None:
        connect = Connection(self.db_path)
        connect.read_sql_script(f"{path_sql_scripts}create_{stg_table_name}.sql")

    def insert_data_to_stg(self, stg_table_name: str, data: pd.DataFrame) -> None:
        connect = Connection(self.db_path)
        connect.insert_data(stg_table_name, data)

    def make_dwh(self, dwh_table_name: str) -> None:
        connect = Connection(self.db_path)
        connect.read_sql_script(dwh_table_name)



