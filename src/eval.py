import pandas as pd


def main():
    print("eval.py is a placeholder.")
    print("Add eval/scenarios.csv and compute purity/mixed-cluster rate here.")
    print("For now, confirm you can load the clustered alerts:")

    df = pd.read_parquet("out/alerts_clustered.parquet")
    print(f"Loaded {len(df)} alerts.")
    print(df[["timestamp", "src_ip", "dest_ip", "signature", "cluster_id", "incident_id"]].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
