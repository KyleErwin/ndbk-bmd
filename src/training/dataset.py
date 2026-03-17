import numpy as np
import pandas as pd


def preprocess_data(filepath):
    df = pd.read_csv(filepath, sep=";")

    df.drop(
        columns=[
            df.columns[0],
            "day",
            "month",
            "contact",
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

    bins = [-2, 0, 7, 30, np.inf]
    labels = ["never", "within_a_week", "within_a_month", "over_a_month"]

    df["pcontacted"] = pd.cut(df["pdays"], bins=bins, labels=labels)
    df.drop(columns=["pdays"], inplace=True)

    num_data = df.select_dtypes(include=np.number)
    cat_data = df.select_dtypes(exclude=np.number)

    num_cols_with_na = num_data.columns[num_data.isnull().any()]
    cat_cols_with_na = cat_data.columns[cat_data.isnull().any()]

    for col in num_cols_with_na:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    for col in cat_cols_with_na:
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)

    cat_features = df.select_dtypes(exclude=np.number).columns.tolist()
    df_encoded = pd.get_dummies(df, columns=cat_features, drop_first=False, dtype=int)

    return df_encoded


def main():
    df = preprocess_data("data/bank_marketing_data.csv")
    df.to_csv("data/bank_marketing_data_processed.csv", index=False)


if __name__ == "__main__":
    main()
