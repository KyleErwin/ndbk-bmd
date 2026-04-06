import numpy as np
import pandas as pd


def preprocess_data(filepath):
    df = pd.read_csv(filepath, sep=";")

    df.drop(
        columns=[
            df.columns[0],
            "day",
            "post_campaign_action",
            "duration",
        ],
        inplace=True,
    )

    df = df.drop_duplicates()

    # Should be illegal to market to someone over the age of 100
    df = df.drop(df[df["age"] > 100].index)

    for x in ["default", "housing", "loan", "target"]:
        df[x] = df[x].str.strip().str.lower().map({"yes": 1, "no": 0})

    month_map = {
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    }
    df["month"] = df["month"].str.lower().map(month_map)

    df["was_contacted"] = (df["pdays"] != -1).astype(int)

    df["negative_balance"] = (df["balance"] < 0).astype(int)
    df["campaign"] = np.log1p(df["campaign"])

    return df


def main():
    df = preprocess_data("data/bank_marketing_data.csv")
    df.to_csv("data/bank_marketing_data_processed.csv", index=False)


if __name__ == "__main__":
    main()
