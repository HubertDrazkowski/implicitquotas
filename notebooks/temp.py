import pandas as pd
import numpy as np
df = pd.read_csv(r"c:\Users\huber\Dropbox\2024_Papers\training\ImplicitQuotas\data\boring.csv")

df = pd.read_csv(r"c:\Users\huber\Dropbox\2024_Papers\training\ImplicitQuotas\data\data.csv")

df = (
    df.sample(frac=0.5,        # keep exactly half the rows
              random_state=42) # any integer seed you like
      .reset_index(drop=True)  # discard the old row numbers
)

unique_labels = df["group"].unique()

# make a random permutation of those unique labels
shuffled_labels = np.random.permutation(unique_labels)

# build a mapping old → new
random_mapping = dict(zip(unique_labels, shuffled_labels))

# apply the mapping
df = df.copy()
df["group"] = df["group"].replace(random_mapping)


df = df.drop("Unnamed: 0", axis=1)
df = df.rename(columns = {"id_bvd": "id_firm"})
df.head(5)

df["year"] = df["year"] + np.random.randint(-2, 3, size=len(df))
df["year"] = df["year"].astype(int).clip(lower=df["year"].min(), upper=df["year"].max())

df["M"] = df["M"] + np.random.randint(0, 5, size=len(df))
df["W"] = df["W"] + np.random.randint(0, 3, size=len(df))
df["M"] = df["M"].astype(int).clip(0, 40)
df["W"] = df["W"].astype(int).clip(0, 40)

df["total"] = df["W"] + df["M"]

noise = np.random.normal(scale=df["supply"].std() * 0.1, size=len(df))
df["supply"] = df["supply"] + noise
df["supply"] = df["supply"].clip(0.0, 100)

df["mainsec"] = (df["mainsec"]
                 .sample(frac=1.0, replace=False, random_state=42)
                 .reset_index(drop=True)
)

df["id"] = (df["id"]
                 .sample(frac=1.0, replace=False, random_state=42)
                 .reset_index(drop=True)
)

df = df.rename(columns = {"mainsec": "group"})
df = df.rename(columns = {"id_firm": "id"})
df.head(5)
print(df.describe())

df.to_csv(r"c:\Users\huber\Dropbox\2024_Papers\training\ImplicitQuotas\data\data.csv", index=False)



