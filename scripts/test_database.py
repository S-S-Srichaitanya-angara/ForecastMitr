import sqlite3
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_PATH = BASE_DIR / "database" / "forecastmitr.db"


connection = sqlite3.connect(DATABASE_PATH)
cursor = connection.cursor()


# Insert test forecast
cursor.execute("""
INSERT INTO forecast (
    latitude,
    longitude,
    issue_time,
    target_time,
    forecast_rainfall
)
VALUES (?, ?, ?, ?, ?)
""", (
    13.0827,
    80.2707,
    "2026-09-12T10:00:00",
    "2026-09-12T18:00:00",
    12.4
))


# Insert test observation
cursor.execute("""
INSERT INTO observation (
    latitude,
    longitude,
    observation_time,
    observed_rainfall
)
VALUES (?, ?, ?, ?)
""", (
    13.0827,
    80.2707,
    "2026-09-12T18:00:00",
    48.7
))


connection.commit()


# Read the data back
print("\nForecast data:")
cursor.execute("SELECT * FROM forecast")

for row in cursor.fetchall():
    print(row)


print("\nObservation data:")
cursor.execute("SELECT * FROM observation")

for row in cursor.fetchall():
    print(row)


connection.close()