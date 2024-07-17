import streamlit as st
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import *

# Inisialisasi SparkSession
spark = SparkSession.builder \
    .appName("Streamlit App") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

# Fungsi untuk mengonversi pandas DataFrame ke Spark DataFrame dengan skema yang ditentukan
def convert_to_spark(df, schema):
    spark_df = spark.createDataFrame(df, schema=schema)
    return spark_df

# Judul aplikasi
st.title("Aplikasi Unggah File Excel atau CSV")

# Unggah file
uploaded_file = st.file_uploader("Pilih file Excel atau CSV", type=["xlsx", "xls", "csv"])

if uploaded_file is not None:
    try:
        # Memeriksa jenis file dan membaca isinya
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Menampilkan isi file
        st.write("Berikut adalah isi file yang diunggah:")
        st.write(df)
        
        # Meminta pengguna untuk mendefinisikan skema untuk setiap kolom
        st.write("Definisikan skema untuk setiap kolom:")
        schema = []
        for col in df.columns:
            col_type = st.selectbox(f"Pilih tipe data untuk kolom {col}", options=['String', 'Integer', 'Float', 'Date'])
            if col_type == 'String':
                schema.append(StructField(col, StringType(), True))
            elif col_type == 'Integer':
                schema.append(StructField(col, IntegerType(), True))
            elif col_type == 'Float':
                schema.append(StructField(col, FloatType(), True))
            elif col_type == 'Date':
                schema.append(StructField(col, DateType(), True))
        
        spark_schema = StructType(schema)
        
        # Mengonversi pandas DataFrame ke Spark DataFrame
        spark_df = convert_to_spark(df, spark_schema)
        
        # Menampilkan Spark DataFrame
        st.write("Berikut adalah Spark DataFrame:")
        st.write(spark_df.show())
        
        # Meminta pengguna untuk memasukkan nama tabel
        table_name = st.text_input("Masukkan nama tabel untuk mengunggah data:")
        
        # Push to database
        if st.button("Unggah ke Database"):
            if table_name:
                # Konfigurasi koneksi database
                db_url = "jdbc:your_database_url"
                db_properties = {
                    "user": "your_username",
                    "password": "your_password",
                    "driver": "your_database_driver"
                }
                
                # Menyimpan DataFrame ke database
                spark_df.write.jdbc(url=db_url, table=table_name, mode='append', properties=db_properties)
                st.success("Data berhasil diunggah ke database!")
            else:
                st.error("Nama tabel tidak boleh kosong.")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat membaca file: {e}")
else:
    st.write("Silakan unggah file untuk melihat isinya.")
