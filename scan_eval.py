import pandas as pd

csv_path = "scan_suspicious.csv"       # <-- change to your file name
num_classes = 8                        # set to 8, 10, etc. if you want a fixed range

df = pd.read_csv(csv_path)

# ---- overall distribution -----------------------------------------------
total = len(df)
dist  = df["class"].value_counts().sort_index()   # Series indexed by class id

print("\n=== Class distribution ===")
print(f"{'class':>5} | {'count':>6} | {'percent':>8}")
print("-" * 28)

for c in range(num_classes):
    n   = dist.get(c, 0)
    pct = 100.0 * n / total if total else 0.0
    print(f"{c:>5} | {n:>6d} | {pct:>7.2f}%")

# Any class IDs outside 0..num_classes-1
extra = dist[~dist.index.isin(range(num_classes))]
if not extra.empty:
    other_n = int(extra.sum())
    other_pct = 100.0 * other_n / total
    print(f"{'other':>5} | {other_n:>6d} | {other_pct:>7.2f}%")

print("-" * 28)
print(f"{'total':>5} | {total:>6d} | {100.00:>7.2f}%")
