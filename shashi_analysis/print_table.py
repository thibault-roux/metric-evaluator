import pandas as pd

df = pd.read_csv("concat_modified.csv")
# Fix dtype
df["accuracy_pct"] = pd.to_numeric(df["accuracy_pct"], errors="coerce")
df["certitude"] = pd.to_numeric(df["certitude"], errors="coerce")

# Filter
cert = 1.0
df_filtered = df[df["certitude"] == cert]

# Pivot
pivot = df_filtered.pivot_table(
    index="model_name",
    columns="pooling",
    values="accuracy_pct"
)

print(pivot)
# print only the name of the model and the accuracy for each pooling method
# save it as a csv file
pivot.to_csv("filtered_results.csv")