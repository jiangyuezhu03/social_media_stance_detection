import json


def clean_text(text):
    return text.strip()

def get_post(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data=json.load(f)
    return data["post"]["content"]

def get_comments_and_replies(filepath):
    # returns a list of strings
    with open(filepath, "r",encoding="utf-8") as f:
        data=json.load(f)
    comments = []
    for c in data["comments"]:
        comments.append(clean_text(c["text"]))
        if "replies" in c:
            for r in c["replies"]:
                comments.append(clean_text(r["text"]))
    return comments
# print(get_comments_and_replies("weibo1.json"))

if __name__ == "__main__":
    import pandas as pd
    from datasets import Dataset
    filepath="c-stance_culture_raw_test_all_onecol.csv"
    df=pd.read_csv(filepath)
    print(df["Type"].unique())
    df_clauses = df.loc[df["Type"] == "clauses"].copy()
    label_map = {"支持": 0, "反对": 1, "中立": 2}
    df_clauses["label"] = df_clauses["Stance 1"].map(label_map)
    df_clauses.drop(columns=['In Use', 'Domain', 'Stance 1', 'Type'], inplace=True)
    df_clauses.rename(columns={"Text": "text", "Target 1": "target"}, inplace=True)
    print(df_clauses.columns)
    df_clauses.to_csv("c-stance_culture_clauses.csv", index=False)
    ds = Dataset.from_pandas(df_clauses)
    ds.remove_columns("__index_level_0__")
    print(ds)
