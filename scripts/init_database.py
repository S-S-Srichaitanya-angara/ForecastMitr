import sqlite3
from pathlib import Path


# Find the root directory of the ForecastMitr project
BASE_DIR = Path(__file__).resolve().parent.parent

# Location where the SQLite database will be stored
DATABASE_DIR = BASE_DIR / "database"
DATABASE_PATH = DATABASE_DIR / "forecastmitr.db"


# Make sure the database directory exists
DATABASE_DIR.mkdir(parents=True, exist_ok=True)


# Connect to SQLite
connection = sqlite3.connect(DATABASE_PATH)

# Cursor allows us to execute SQL commands
cursor = connection.cursor()


# --------------------------------------------------
# Forecast table
# --------------------------------------------------

cursor.execute("""
CREATE TABLE IF NOT EXISTS forecast (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    issue_time TEXT NOT NULL,
    target_time TEXT NOT NULL,
    forecast_rainfall REAL
)
""")


# --------------------------------------------------
# Observation table
# --------------------------------------------------

cursor.execute("""
CREATE TABLE IF NOT EXISTS observation (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    observation_time TEXT NOT NULL,
    observed_rainfall REAL
)
""")


# Save changes
connection.commit()

# Close database connection
connection.close()


print("Database initialized successfully!")
print(f"Database location: {DATABASE_PATH}")