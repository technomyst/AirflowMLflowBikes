import sqlite3
import pandas as pd
import logging
logging.basicConfig(level=logging.DEBUG, filename='myapp.log', format='%(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class Connection:

    def __init__(self, conn_path: str) -> None:
        self.conn_path = conn_path

    def _create_connection(self):
        return sqlite3.connect(self.conn_path)

    @staticmethod
    def open_file(file_path: str) -> str:
        with open(file_path, 'r') as f:
            file = f.read()
        return file

    def read_sql_script(self, sql_path: str, file_script: bool = True) -> None:
        """
        """
        conn = self._create_connection()
        try:
            cur = conn.cursor()
            if file_script:
                sql = self.open_file(sql_path)
            else:
                sql = sql_path
            cur.executescript(sql)
            conn.commit()
        except Exception as error:
            print(error)
            logger.info(error)
        finally:
            conn.close()

    def save_select_result_to_csv(self, table_name: str,output_path,filename) -> None:

        conn = self._create_connection()
        try:
            db_df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            logger.info(db_df)
            db_df.to_csv(f'{output_path}{filename}')
        except Exception as error:
            print(error)
            logger.info(error)
        finally:
            conn.close()

    def insert_data(self, table: str, data: pd.DataFrame) -> None:
        """
        insert data into stg_table
        :param data:
        :param table:
        :return: None
        """
        data = data.astype('str')
        conn = self._create_connection()

        query = f"""INSERT INTO {table}({",".join(list(data))})
                VALUES ({",".join(['?' for x in range(data.shape[1])])})"""
        try:
            cur = conn.cursor()
            cur.executemany(query, list(data.to_records(index=False)))
            conn.commit()
        except Exception as error:
            print(error)
            logger.info(error)
        finally:
            conn.close()