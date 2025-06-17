import psycopg2
import os
from dotenv import load_dotenv
load_dotenv()

conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    database=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
)
cur = conn.cursor()
cur.execute("SELECT id, email, created_at FROM users;")
rows = cur.fetchall()
print("ユーザー一覧：")
for row in rows:
    print(f"id: {row[0]}, email: {row[1]}, 登録日: {row[2]}")
cur.close()
conn.close()
