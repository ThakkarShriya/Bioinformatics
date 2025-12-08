import os
import shutil
import pandas as pd

metadata_path = "oasis_cross-sectional-5708aa0a98d82080.xlsx"
source_base = "/Users/shriyathakkar/Downloads/disc1"
dest_healthy = "/Users/shriyathakkar/Desktop/dementia_pca_project/dataset/healthy"
dest_mci = "/Users/shriyathakkar/Desktop/dementia_pca_project/dataset/mci"

df = pd.read_excel(metadata_path, sheet_name="oasis_cross-sectional")
df = df.dropna(subset=["CDR"])  # Only subjects with a known diagnosis

copied = 0
missing = 0

for _, row in df.iterrows():
    subject_id = row["ID"]
    cdr = row["CDR"]

    if cdr not in [0.0, 0.5]:
        continue

    gif_name = f"{subject_id}_mpr_n4_anon_111_t88_masked_gfc_fseg_tra_90.gif"
    src = os.path.join(source_base, subject_id, "FSL_SEG", gif_name)

    if cdr == 0.0:
        dst = os.path.join(dest_healthy, f"{subject_id}.gif")
    else:
        dst = os.path.join(dest_mci, f"{subject_id}.gif")

    if os.path.exists(src):
        shutil.copy(src, dst)
        copied += 1
    else:
        print(f"❌ Missing file: {src}")
        missing += 1

print(f"\n✅ Done! {copied} files copied. {missing} missing.")
