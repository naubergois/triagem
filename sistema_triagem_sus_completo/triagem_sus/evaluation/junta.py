# Merge multiple JSON files in /mnt/data into a single dataset
import os, json, glob
import pandas as pd


input_pattern = "*.json"
files = sorted(glob.glob(input_pattern))
print(files)
records = []
for fp in files:
    try:
        with open(fp, "r", encoding="utf-8") as f:
            obj = json.load(f)
        # Attach source file name for traceability
        obj["_source_file"] = os.path.basename(fp)
        records.append(obj)
    except Exception as e:
        print(f"Erro ao ler {fp}: {e}")

# Create output directory
out_dir = "./output"
os.makedirs(out_dir, exist_ok=True)

# Save as JSON array
json_array_path = os.path.join(out_dir, "triagem_merged.json")
with open(json_array_path, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

# Save as JSON Lines
jsonl_path = os.path.join(out_dir, "triagem_merged.jsonl")
with open(jsonl_path, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# Also provide CSV for convenience
df = pd.DataFrame.from_records(records)
csv_path = os.path.join(out_dir, "triagem_merged.csv")
df.to_csv(csv_path, index=False, encoding="utf-8")

# And Parquet (if fields are mostly tabular)
parquet_path = os.path.join(out_dir, "triagem_merged.parquet")
df.to_parquet(parquet_path, index=False)


