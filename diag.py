import sys
sys.path.insert(0, "src/ingestion")
from cpcb_client import fetch_current_aqi

df = fetch_current_aqi(limit=2000)
print("total rows returned:", len(df))
if len(df):
    print(df["state"].value_counts().head(10))
    print("TN cities:", df[df["state"].str.contains("Tamil", na=False)]["city"].unique())