import json
import pendulum
from airflow.decorators import dag, task
from config import db, root_path, root_path_input_data, root_path_archive, tables_info,path_sql_scripts,output_data,select_dwh_for_model_script
import logging
logging.basicConfig(level=logging.DEBUG, filename='myapp.log', format='%(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

@dag(
    schedule=None,
    start_date=pendulum.datetime(2021, 1, 1, tz="UTC"),
    catchup=False,
    tags=["etl", "classification"],
)
def item_classification(debug=False):

    @task()
    def extract_data_from_files_to_stage():
        from stages.etl.connection import Connection
        from stages.etl.extract_data import Extract_data

        conn = Connection(db)._create_connection()
        extract = Extract_data(root_path, conn, root_path_input_data, root_path_archive,tables_info,db,path_sql_scripts)
        result=extract.my_extract_data_to_stg()
        '''table = extract.create_tabel('items')'''
        '''return table'''
        return f"{result}"

    @task()
    def transfer_filesdata_from_stage_to_dwh(stg_result):
        from stages.etl.connection import Connection
        from stages.etl.extract_data import Extract_data

        conn = Connection(db)._create_connection()
        extract = Extract_data(root_path, conn, root_path_input_data, root_path_archive, tables_info, db,
                               path_sql_scripts)
        result = extract.my_extract_filesdata_from_stg_to_dwh()

        '''table = extract.create_tabel('items')'''
        '''return table'''
        return f"{result}"

    @task()
    def extract_web_data(categories_and_webfilters):
        from stages.parsing.parcer import Parcer
        from stages.etl.connection import Connection
        conn = Connection(db)._create_connection()
        par = Parcer(*categories_and_webfilters)
        data = par.take_website_data()
        table_name = 'STG_'+'_'.join(categories_and_webfilters)
        logger.info(table_name)
        data.to_sql(table_name, con=conn, if_exists='replace')
        return table_name

    @task()
    def create_dwh_parcer_data(list_table):
        from stages.sql_scripts.union_tables import union_tabels
        from stages.etl.connection import Connection
        conn = Connection(db)
        logger.info(list_table)
        tables = union_tabels(list_table)
        s = f"""
        Create table if not EXISTS DWH_ProductsStat as {tables}
        """
        logger.info(s)
        conn.read_sql_script(s, file_script=False)
        return 'DWH_ProductsStat'

    @task()
    def create_outputfile_for_dashboards(dwh_table_for_output_dashboard):
        from stages.sql_scripts.union_tables import union_tabels
        from stages.etl.connection import Connection
        conn = Connection(db)
        conn.save_select_result_to_csv("DWH_ProductsStat",output_data,"ProductsStat.csv")
        return 'ProductsStat.csv'

    @task()
    def preprocessing(dwh_tables):
        from stages.sql_scripts.union_tables import union_tabels
        from stages.etl.connection import Connection
        import pandas as pd
        from stages.etl.preprocessing import Preprocessing

        print(pd.__version__)
        conn = Connection(db)._create_connection()
        data = pd.read_sql_query(select_dwh_for_model_script, con=conn)
        cat_f = ['gender', 'job_title','job_industry_category','wealth_segment','state','brand','class','line','size','address','country']
        process = Preprocessing(cat_f)
        process_data = process.process_data(data)
        logger.info(process_data.columns)
        process_data.to_sql('model_data', con=conn, if_exists='replace')
        return 'model_data'

    @task(multiple_outputs=True)
    def train_test_split(model_data):
        from stages.etl.connection import Connection
        import pandas as pd
        from sklearn.model_selection import train_test_split
        conn = Connection(db)._create_connection()
        data = pd.read_sql_query(f"""Select * from {model_data}""", con=conn)
        logger.info(data.columns)
        train_data, test_data = train_test_split(data, random_state=12345, test_size=0.2)
        logger.info(train_data.columns)
        logger.info(test_data.columns)
        train_data.to_sql('train_data', con=conn, if_exists='replace')
        test_data.to_sql('test_data', con=conn, if_exists='replace')
        return {'test_data': 'test_data', 'train_data': 'train_data'}

    @task(multiple_outputs=True)
    def model_materialize(model_name, test_data):
        import mlflow
        from stages.etl.connection import Connection
        import pandas as pd
        import os

        d = dict(
            AWS_ACCESS_KEY_ID="admin",
            AWS_SECRET_ACCESS_KEY="sample_key",
            AWS_REGION="us - east - 1",
            AWS_BUCKET_NAME="mlflow",
            MYSQL_DATABASE="mlflow",
            MYSQL_USER="mlflow_user",
            MYSQL_PASSWORD="mlflow_password",
            MYSQL_ROOT_PASSWORD="toor",
            MLFLOW_TRACKING_URI='http://mlflow:5001',
            MLFLOW_EXPERIMENT_NAME="MyExp",
            MLFLOW_S3_ENDPOINT_URL='http://s3:9000'
        )

        for i in d:
            os.environ[i] = d[i]

        logged_model = f'models:/{model_name}/None'
        conn = Connection(db)._create_connection()
        # Load model as a PyFuncModel.
        loaded_model = mlflow.pyfunc.load_model(logged_model)
        test = pd.read_sql_query(f"""Select * from {test_data}""", con=conn)
        predict = loaded_model.predict(test.drop(columns=['order_status_int']))
        test['predict'] = predict
        test[['customer_id','product_id', 'predict']].to_sql('model_predict', con=conn, if_exists='replace')

    @task()
    def get_metrics(model_name, test_data):
        import mlflow
        from stages.etl.connection import Connection
        from stages.models.linear_model import LinearModel
        import pandas as pd
        import os

        d = dict(
            AWS_ACCESS_KEY_ID="admin",
            AWS_SECRET_ACCESS_KEY="sample_key",
            AWS_REGION="us - east - 1",
            AWS_BUCKET_NAME="mlflow",
            MYSQL_DATABASE="mlflow",
            MYSQL_USER="mlflow_user",
            MYSQL_PASSWORD="mlflow_password",
            MYSQL_ROOT_PASSWORD="toor",
            MLFLOW_TRACKING_URI='http://mlflow:5001',
            MLFLOW_EXPERIMENT_NAME="MyExp",
            MLFLOW_S3_ENDPOINT_URL='http://s3:9000'
        )

        for i in d:
            os.environ[i] = d[i]

        logged_model = f'models:/{model_name}/None'
        conn = Connection(db)._create_connection()
        # Load model as a PyFuncModel.
        loaded_model = mlflow.sklearn.load_model(logged_model)
        mmodel = LinearModel()
        mmodel.model = loaded_model
        test = pd.read_sql_query(f"""Select * from {test_data}""", con=conn)
        metrics = mmodel.get_metrics(test.drop(columns=['order_status_int','index','level_0']), test['order_status_int'])

        for j in metrics:
            mlflow.log_metric(j, metrics[j])

    @task()
    def model_fit(train_data, test_data):
        import os
        from stages.etl.connection import Connection
        import pandas as pd
        from stages.models.linear_model import LinearModel
        import mlflow
        from mlflow.models import infer_signature
        d = dict(
            AWS_ACCESS_KEY_ID="admin",
            AWS_SECRET_ACCESS_KEY="sample_key",
            AWS_REGION="us - east - 1",
            AWS_BUCKET_NAME="mlflow",
            MYSQL_DATABASE="mlflow",
            MYSQL_USER="mlflow_user",
            MYSQL_PASSWORD="mlflow_password",
            MYSQL_ROOT_PASSWORD="toor",
            MLFLOW_TRACKING_URI='http://mlflow:5001',
            MLFLOW_EXPERIMENT_NAME="MyExp",
            MLFLOW_S3_ENDPOINT_URL='http://s3:9000'
        )

        for i in d:
            os.environ[i] = d[i]

        conn = Connection(db)._create_connection()
        train = pd.read_sql_query(f"""Select * from {train_data}""", con=conn)
        logger.info('train_data')
        logger.info(train.columns)

        model = LinearModel()
        with mlflow.start_run() as run:
            model.fit(train.drop(columns=['order_status_int','level_0', 'index']), train['order_status_int'])

            test = pd.read_sql_query(f"""Select * from {test_data}""", con=conn)
            logger.info('test_data')
            logger.info(train.columns)
            preds = model.model.predict(test.drop(columns=['order_status_int','level_0', 'index']))

            signature = infer_signature(train.drop(columns=['order_status_int','level_0', 'index']), preds)
            mlflow.sklearn.log_model(model.model, "model", signature=signature, registered_model_name='LinearModel')

        return 'LinearModel'

    stg_file_result = extract_data_from_files_to_stage()
    dwh_file_result = transfer_filesdata_from_stage_to_dwh(stg_file_result)
    item_category = ['shlemy', 'flyagi', 'ryukzaki_i_sumki']
    item_filter = ['new', 'discount', 'popular']
    list3 = [[i, str(j)] for i in item_category for j in item_filter]
    website_tables = extract_web_data.expand(categories_and_webfilters=list3)
    dwh_table_website = create_dwh_parcer_data(website_tables)
    output_dashboard_file = create_outputfile_for_dashboards(dwh_table_website)

    prep = preprocessing(dwh_file_result)
    train_test = train_test_split(prep)
    model = model_fit(train_test['train_data'], train_test['test_data'])
    model_mat = model_materialize(model, train_test['test_data'])
    get_metrics(model, train_test['test_data'])
prod = item_classification(debug=False)
