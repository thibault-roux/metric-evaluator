import csv

INPUT_FILE = "hats-en.txt"

with open(INPUT_FILE, "r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f, delimiter="\t")

    for line_num, row in enumerate(reader, start=2):
        for field in ("reference", "hypA", "hypB"):
            value = row.get(field, "").strip()

            if len(value) < 1:
                print(f"EMPTY {field} at line {line_num}")

            # If you mean less than 2 characters:
            # if len(value) < 2:
            #     print(f"TOO SHORT {field} at line {line_num}: {value!r}")
