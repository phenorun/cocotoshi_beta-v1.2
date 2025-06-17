import psycopg2
import os
from dotenv import load_dotenv
load_dotenv()

conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    database=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD")
)
cur = conn.cursor()
cur.execute("ALTER TABLE users ALTER COLUMN password_hash TYPE VARCHAR(512);")
conn.commit()
cur.close()
conn.close()
print("password_hashカラムの長さを512に拡張しました！")
