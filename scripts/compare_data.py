import sqlite3
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_PATH = BASE_DIR / "database" / "forecastmitr.db"


connection = sqlite3.connect(DATABASE_PATH)
cursor = connection.cursor()


query = """
SELECT
    f.latitude,
    f.longitude,
    f.issue_time,
    f.target_time,
    f.forecast_rainfall,
    o.observed_rainfall
FROM forecast AS f
JOIN observation AS o
    ON f.latitude = o.latitude
    AND f.longitude = o.longitude
    AND f.target_time = o.observation_time
"""


cursor.execute(query)

rows = cursor.fetchall()


for row in rows:
    latitude, longitude, issue_time, target_time, forecast, observed = row

    print()
    print("Location:", latitude, longitude)
    print("Forecast issued:", issue_time)
    print("Target time:", target_time)
    print("Forecast rainfall:", forecast, "mm")
    print("Observed rainfall:", observed, "mm")
    print("Forecast error:", abs(observed - forecast), "mm")


connection.close()