import sqlite3
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_PATH = BASE_DIR / "database" / "forecastmitr.db"


connection = sqlite3.connect(DATABASE_PATH)
cursor = connection.cursor()


# Get all tables
cursor.execute("""
SELECT name
FROM sqlite_master
WHERE type='table'
""")

tables = cursor.fetchall()

print("Tables in database:")

for table in tables:
    print("-", table[0])


connection.close()