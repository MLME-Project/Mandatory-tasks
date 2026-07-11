import os
import pandas as pd
from API_Group8 import BioreactorClient

COLUMNS = ["id", "created_at", "scale", "T", "pH", "F1", "F2", "F3", "Y", "cost_eur"]
INPUT_COLS = ["T", "pH", "F1", "F2", "F3"]

def getDataFrameFromCSV(fileName):
    return pd.read_csv(fileName, index_col="id")

def getDataFrameFromClient():
    client = BioreactorClient()
    client.login()
    items = client.history()
    rows = [{
        "id": item["id"],
        "created_at": item["created_at"],
        "scale": item["scale"],
        **item["recipe"],
        "Y": item["Y"],
        "cost_eur": item["cost_eur"],
    } for item in items]
    return pd.DataFrame(rows, columns=COLUMNS).set_index("id")

def createCSV(fileName):
    pd.DataFrame(columns=COLUMNS).set_index("id").to_csv(fileName)

def appendToCSV(fileName, scale, T, pH, F1, F2, F3, result):
    row = pd.DataFrame([{
        "id": result["id"],
        "created_at": result["created_at"],
        "scale": scale,
        "T": T, "pH": pH, "F1": F1, "F2": F2, "F3": F3,
        "Y": result["Y"],
        "cost_eur": result["cost_eur"],
    }], columns=COLUMNS).set_index("id")

    if os.path.exists(fileName):
        # match any extra columns (e.g. measured_pilot_Y) added to the file since creation
        existing_cols = pd.read_csv(fileName, index_col="id", nrows=0).columns
        row = row.reindex(columns=existing_cols)
        write_header = False
    else:
        write_header = True

    row.to_csv(fileName, mode="a", header=write_header)

def getXyFromDataFrame(df):
    return df[INPUT_COLS].values, df["Y"].values

def getXyFromCSV(fileName):
    return getXyFromDataFrame(getDataFrameFromCSV(fileName=fileName))

# df = getDataFrameFromClient()
# df = getDataFrameFromCSV('mlme26_group08_2026-06-29T10-47-17.csv')
# print(df)