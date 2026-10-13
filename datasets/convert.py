import pandas as pd

df = pd.read_csv("hats-en.csv")

# Optional filters:
# df = df[df["batch"] == "01"]
# df = df[df["triplet_id"] == "set_01_121-121726-0000.wav"]

with open("hats-en.txt", "w", encoding="utf-8") as f:
    f.write("reference\thypA\tnbrA\thypB\tnbrB\n")
    for _, row in df.iterrows():
        f.write(
            f'{row["reference"]}\t'
            f'{row["h1_text"]}\t'
            f'{row["h1_votes"]}\t'
            f'{row["h2_text"]}\t'
            f'{row["h2_votes"]}\n'
        )
