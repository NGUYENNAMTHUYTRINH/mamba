import pandas as pd
from collections import Counter

def main():
    path = "dataset/2025.csv"
    df = pd.read_csv(path, usecols=["location_key"]) if True else None
    s = df["location_key"].astype(str).dropna()
    counts = Counter(s.tolist())
    top = counts.most_common(10)
    print("Total unique locations:", len(counts))
    for loc, c in top:
        print(f"{loc}\t{c}")

if __name__ == '__main__':
    main()
