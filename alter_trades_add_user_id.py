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
cur.execute("""
    ALTER TABLE trades
    ADD COLUMN IF NOT EXISTS user_id INTEGER;
""")
conn.commit()
cur.close()
conn.close()
print("user_idカラム追加、成功！")
