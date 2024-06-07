import pyodbc
from SimeCSE_Vn_model import preprocess_text, embedding_text

connection_string = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    f"SERVER={'10.0.0.39'};"
    f"DATABASE={'DataV03'};"
    f"UID={'sa'};"
    f"PWD={'Ab@123456'}"
)

def data_sql():
    try:
        connection = pyodbc.connect(connection_string)
        cursor = connection.cursor()

        table_exists_query = """
            SELECT * FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_NAME = 'data_vecter'
            """

        cursor.execute(table_exists_query)
        table_info = cursor.fetchone()
        if table_info is None:
            create_table_query = """
                CREATE TABLE data_vecter (
                    ID INT,
                    SoHieu NVARCHAR(200),
                    LoaiVanBan NVARCHAR(200),
                    NoiBanHanh NVARCHAR(1000),
                    NguoiKy NVARCHAR(1000),
                    NgayBanHanh NVARCHAR(100),
                    NgayHieuLuc NVARCHAR(100),
                    NgayCongBao NVARCHAR(100),
                    SoCongBao NVARCHAR(100),
                    TinhTrang NVARCHAR(100),
                    Chuong INT,
                    Muc INT,
                    Dieu_or_ChuyenMuc INT,
                    Vecter NVARCHAR(max)
                );
            """

            cursor.execute(create_table_query)
            connection.commit()
            print("Table 'data_vecter' created successfully.")
        else:
            print("Table 'data_vecter' already exists.")
            
        sql_query = "SELECT TOP 1000 * FROM [DataV03].[dbo].[data_ND]"  # Replace with your actual query

        cursor.execute(sql_query)
        rows = cursor.fetchall()
        data_json_new = []
        for row in rows:
            ids = int(row[0])
            jsons = {
                "id-van-ban": ids,
                "so-hieu": row[1],
                "loai-van-ban": row[2],
                "noi-ban-hanh": row[3],
                "nguoi-ky": row[4],
                "ngay-ban-hanh": row[5],
                "ngay-hieu-luc": row[6],
                "ngay-cong-bao": row[7],
                "so-cong-bao": row[8],
                "tinh-trang": row[9],
                "chuong": row[10],
                "muc": row[11],
                "dieu": row[12],
                "noi-dung": row[13]
            }
            data_json_new.append(jsons)
        return data_json_new
    except pyodbc.Error as ex:
        print("Error connecting to SQL Server:", ex)

    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()