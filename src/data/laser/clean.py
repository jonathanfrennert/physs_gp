# Downloaded from https://data.gulfresearchinitiative.org/data/R4.x265.237:0001
from pathlib import Path
import pandas as pd

output_path = Path('data')
output_path.mkdir(exist_ok=True)

raw_data = Path('R4.x265.237.0001') / 'laser_spot_drifters_clean_v15.dat'

df = pd.read_csv(raw_data, header=None, delim_whitespace=True, skiprows=5)
df.columns = [
    'id',
    'date',
    'time',
    'lat', 
    'lon',
    'position_error',
    'u',
    'v',
    'velocity_error'
]

df['datetime_micro'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y-%m-%d %H:%M:%S.%f')

# round to 15 mins
df['datetime'] = df['datetime_micro'].dt.floor('S')

print(f"saving to { output_path / 'laser.csv'}")

df.to_csv(
    output_path / 'laser.csv',
    index=False
)
