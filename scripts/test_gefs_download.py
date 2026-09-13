from pathlib import Path
import requests


# ============================================================
# CONFIGURATION
# ============================================================

BASE_URL = (
    "https://nomads.ncep.noaa.gov/"
    "cgi-bin/filter_gefs_atmos_0p25s.pl"
)

DATE = "20260913"
CYCLE = "00"

# Chennai region.
# Slightly larger than one grid point so we can verify
# the geographic subset.
LEFT_LON = 79.75
RIGHT_LON = 80.75
TOP_LAT = 13.75
BOTTOM_LAT = 12.75

# We will request APCP.
# The exact file name will be supplied by NOMADS after
# we inspect the available files.