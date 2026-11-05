
# Library Import
import json
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

# Importing Dataset
file_path = DATA_DIR / "T_ONTIME_REPORTING.csv"
df = pd.read_csv(file_path)

# Preserve the BTS destination ID that corresponds to each IATA input.
destinations = df.loc[df['ORIGIN'].eq('ATL'), ['DEST', 'DEST_AIRPORT_ID']].drop_duplicates()
if not destinations['DEST'].is_unique or not destinations['DEST_AIRPORT_ID'].is_unique:
    raise ValueError("Conflicting airport identifiers for Atlanta departures")
iata_to_bts = dict(zip(destinations['DEST'], destinations['DEST_AIRPORT_ID'].astype(str)))
with (ROOT / "src" / "iata_to_bts.json").open("w") as f:
    json.dump(iata_to_bts, f, indent=2, sort_keys=True)
    f.write("\n")

# Display df
df.head(10)

# Column Formatting & Filtering

# | YEAR | MONTH | DAY | DAY_OF_WEEK | ORG_AIRPORT | DEST_AIRPORT | SCHEDULED_DEPARTURE | DEPARTURE_TIME | DEPARTURE_DELAY | SCHEDULED_ARRIVAL | ARRIVAL_TIME | ARRIVAL_DELAY |
# |:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
# | integer | integer | integer | integer | string | string | integer | integer | integer | integer | integer | integer |


# Select only the required columns and rename them
df = df.rename(columns={
    'DAY_OF_MONTH': 'DAY',
    'ORIGIN_AIRPORT_ID': 'ORG_AIRPORT',
    'DEST_AIRPORT_ID': 'DEST_AIRPORT',
    'CRS_DEP_TIME': 'SCHEDULED_DEPARTURE',
    'DEP_TIME': 'DEPARTURE_TIME',
    'DEP_DELAY': 'DEPARTURE_DELAY',
    'CRS_ARR_TIME': 'SCHEDULED_ARRIVAL',
    'ARR_TIME': 'ARRIVAL_TIME',
    'ARR_DELAY': 'ARRIVAL_DELAY'
})[['YEAR', 'MONTH', 'DAY', 'DAY_OF_WEEK', 'ORG_AIRPORT', 'DEST_AIRPORT', 
     'SCHEDULED_DEPARTURE', 'DEPARTURE_TIME', 'DEPARTURE_DELAY', 
     'SCHEDULED_ARRIVAL', 'ARRIVAL_TIME', 'ARRIVAL_DELAY']]


# Convert all columns except ORG_AIRPORT and DEST_AIRPORT to integer type
# df = df.apply(lambda col: col.astype('Int64') if col.name not in ['ORG_AIRPORT', 'DEST_AIRPORT'] else col)

# Check data types of each column
print(df.dtypes)

print(df.head())

# Filtering by Origin Airport (ATL)
# Dest Airport ID: 10397

df_atl = df[df['ORG_AIRPORT'] == 10397].copy()

# Display the filtered dataframe
print(df_atl.head())

df = df_atl

# Check for missing values by column
missing_values = df.isnull().sum()

print(missing_values)

# Dropping rows with missing values in specific columns
df.dropna(subset=['DEPARTURE_TIME', 'DEPARTURE_DELAY', 'ARRIVAL_TIME', 'ARRIVAL_DELAY'], inplace=True)

#Ensuring null data is removed
missing_values = df.isnull().sum()

print(missing_values)

# Convert ORG_AIRPORT and DEST_AIRPORT to string type
# MLFlow has issues with integer types for these columns - deprecated typing"
df[['ORG_AIRPORT', 'DEST_AIRPORT']] = df[['ORG_AIRPORT', 'DEST_AIRPORT']].astype(str)

# Verify the changes
print(df.dtypes)

# Identify duplicate rows
duplicates = df.duplicated()

# Count the number of duplicate rows
num_duplicates = duplicates.sum()
print(f"Number of duplicate rows: {num_duplicates}")

# Display duplicate rows if any
if num_duplicates > 0:
    # Display the duplicate rows before dropping them
    print(df[duplicates].head())
    
    # Drop duplicate rows
    df.drop_duplicates(inplace=True)
    
    # Recalculate the number of duplicate rows
    new_num_duplicates = df.duplicated().sum()
    
    # Print the number of duplicates removed
    print(f"Number of duplicate rows removed: {num_duplicates - new_num_duplicates}")

# Dropping spaces before and after each cell
df = df.apply(lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x))

# Save the dataframe to a CSV file
df.to_csv(DATA_DIR / 'cleaned_data.csv', index=False)
