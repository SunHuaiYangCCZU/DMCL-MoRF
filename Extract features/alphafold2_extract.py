from Bio import SeqIO
from pathlib import Path
import pandas as pd
import json
import numpy as np


rows = []

for idx, record in enumerate(SeqIO.parse(fasta_path, "fasta")):
    full_id = record.id
    seq_name = full_id.split("|")[0]
    sequence = str(record.seq)

    pdb_path = af_out_dir / seq_name / "ranked_0.pdb"
    result_jsons = list((af_out_dir / seq_name).glob("result_model_*.json"))

    if not pdb_path.exists() or len(result_jsons) == 0:
        print(f"[WARN] {seq_name} missing ranked_0 or result json")
        rows.append({
            "order": idx,
            "id": full_id,
            "sequence": sequence,
            "mean_plddt": None
        })
        continue

    with open(result_jsons[0]) as f:
        data = json.load(f)

    mean_plddt = float(np.mean(data["plddt"]))

    rows.append({
        "order": idx,
        "id": full_id,
        "sequence": sequence,
        "mean_plddt": mean_plddt
    })

df = pd.DataFrame(rows)
df = df.sort_values("order")
df.to_csv("alphafold_ranked0.csv", index=False)
