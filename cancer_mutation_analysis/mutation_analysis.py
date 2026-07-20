import pandas as pd
import numpy as np
from scipy import stats
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns

# Load mutation data
lgg = pd.read_csv("data_mutations_lgg.txt", sep="\t", comment="#", low_memory=False)
gbm = pd.read_csv("data_mutations_gbm.txt", sep="\t", comment="#", low_memory=False)

# Combine into one pan-glioma dataset
df = pd.concat([lgg, gbm], ignore_index=True)

# Quick summary
print("LGG mutations:", len(lgg))
print("GBM mutations:", len(gbm))
print("Total mutations:", len(df))
print("Columns:", list(df.columns))

# Keep only the columns we need
cols = [
    "Hugo_Symbol",           # gene name
    "Variant_Classification",# mutation type (Missense, Nonsense, etc.)
    "Variant_Type",          # SNP, INS, DEL
    "Tumor_Sample_Barcode",  # patient ID
    "HGVSp_Short",           # amino acid change
    "t_depth",               # tumor read depth (quality indicator)
    "IMPACT"                 # mutation impact (HIGH, MODERATE, LOW)
]

df = df[cols]

print("\nCleaned dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nMutation types found:")
print(df["Variant_Classification"].value_counts())

# Step 1: Remove low quality mutations (tumor read depth below 10)
df = df[df["t_depth"] >= 10]
print("After quality filter:", len(df))

# Step 2: Remove Silent mutations (these are passengers by definition)
df = df[df["Variant_Classification"] != "Silent"]
print("After removing Silent mutations:", len(df))

# Step 3: Keep only the most relevant mutation types for driver analysis
driver_relevant = [
    "Missense_Mutation",
    "Nonsense_Mutation",
    "Frame_Shift_Del",
    "Frame_Shift_Ins",
    "Splice_Site",
    "In_Frame_Del",
    "In_Frame_Ins",
    "Translation_Start_Site",
    "Nonstop_Mutation"
]

df = df[df["Variant_Classification"].isin(driver_relevant)]
print("After keeping driver-relevant mutations:", len(df))

# Step 4: Remove rows where gene name is missing
df = df.dropna(subset=["Hugo_Symbol"])
print("After removing missing gene names:", len(df))

# Final summary
print("\nFinal cleaned dataset shape:", df.shape)
print("\nMutation types remaining:")
print(df["Variant_Classification"].value_counts())

# Step 2: Mutation Frequency Analysis

# Count total number of unique patients
total_patients = df["Tumor_Sample_Barcode"].nunique()
print("Total patients:", total_patients)

# Count how many unique patients have a mutation in each gene
gene_mutation_counts = df.groupby("Hugo_Symbol")["Tumor_Sample_Barcode"].nunique()

# Convert to a dataframe and sort by most mutated
gene_mutation_counts = gene_mutation_counts.reset_index()
gene_mutation_counts.columns = ["Gene", "Patients_Mutated"]

# Calculate mutation frequency as a percentage of patients
gene_mutation_counts["Mutation_Frequency"] = (
    gene_mutation_counts["Patients_Mutated"] / total_patients * 100
).round(2)

# Sort by frequency
gene_mutation_counts = gene_mutation_counts.sort_values(
    "Mutation_Frequency", ascending=False
).reset_index(drop=True)

# Show top 20 most mutated genes
print("\nTop 20 most mutated genes:")
print(gene_mutation_counts.head(20).to_string(index=False))

# Step 3: Binomial Test for Driver Mutation Detection

from scipy.stats import binomtest

# Background mutation rate — the baseline probability that any given gene
# is mutated in a patient by random chance alone
# This is a standard estimate used in cancer genomics literature
background_rate = 0.05

# Run binomial test for every gene
results = []

for _, row in gene_mutation_counts.iterrows():
    gene = row["Gene"]
    mutated_patients = int(row["Patients_Mutated"])
    
    # Test: is this gene mutated in significantly more patients
    # than we would expect by random chance?
    result = binomtest(mutated_patients, total_patients, background_rate, alternative="greater")
    
    results.append({
        "Gene": gene,
        "Patients_Mutated": mutated_patients,
        "Mutation_Frequency": row["Mutation_Frequency"],
        "P_Value": result.pvalue
    })

# Convert to dataframe
results_df = pd.DataFrame(results)

# Apply Bonferroni correction for multiple testing
# (we tested thousands of genes so we need to raise the significance bar)
num_genes = len(results_df)
bonferroni_threshold = 0.05 / num_genes

# Label each gene as driver or passenger
results_df["Significant"] = results_df["P_Value"] < bonferroni_threshold
results_df["Label"] = results_df["Significant"].map({True: "Driver", False: "Passenger"})

# Sort by p-value
results_df = results_df.sort_values("P_Value").reset_index(drop=True)

print(f"Total genes tested: {num_genes}")
print(f"Bonferroni threshold: {bonferroni_threshold:.6f}")
print(f"\nDriver genes found: {results_df['Significant'].sum()}")
print(f"Passenger genes: {(~results_df['Significant']).sum()}")
print("\nTop 20 statistically significant driver candidates:")
print(results_df[results_df["Label"] == "Driver"].head(20).to_string(index=False))

# Step 4: Machine Learning Model to Predict Driver vs Passenger

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Step 4a: Build features for each gene
gene_features = df.groupby("Hugo_Symbol").apply(lambda x: pd.Series({
    "Mutation_Frequency": x["Tumor_Sample_Barcode"].nunique() / total_patients * 100,
    "High_Impact_Rate": (x["IMPACT"] == "HIGH").mean(),
    "Missense_Rate": (x["Variant_Classification"] == "Missense_Mutation").mean(),
    "Nonsense_Rate": (x["Variant_Classification"] == "Nonsense_Mutation").mean(),
    "Frameshift_Rate": (x["Variant_Classification"].isin(
        ["Frame_Shift_Del", "Frame_Shift_Ins"])).mean(),
    "Total_Mutations": len(x)
})).reset_index()

# Step 4b: Merge binomial test labels into gene features
gene_features = gene_features.merge(
    results_df[["Gene", "Label", "P_Value"]],
    left_on="Hugo_Symbol",
    right_on="Gene",
    how="left"
).drop(columns=["Gene"])

gene_features = gene_features.dropna(subset=["Label"])

print("Gene feature table shape:", gene_features.shape)
print("\nDriver vs Passenger counts:")
print(gene_features["Label"].value_counts())

# Step 4c: Prepare data for training
feature_cols = [
    "Mutation_Frequency",
    "High_Impact_Rate",
    "Missense_Rate",
    "Nonsense_Rate",
    "Frameshift_Rate",
    "Total_Mutations"
]

X = gene_features[feature_cols]
y = gene_features["Label"]

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Step 4d: Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

# Step 4e: Evaluate the model
y_pred = model.predict(X_test)

print("\nModel Performance:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Step 4f: Feature importance
print("\nFeature Importance:")
importance_df = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)
print(importance_df.to_string(index=False))

# Step 5: Visualizations for Poster

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set global style
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150

# ── Graph 1: Top 20 Most Mutated Genes ──────────────────────────────────────
top20 = gene_mutation_counts.head(20).copy()
top20["Label"] = top20["Gene"].apply(
    lambda g: "Driver" if g in results_df[results_df["Label"] == "Driver"]["Gene"].values else "Passenger"
)

plt.figure(figsize=(12, 7))
colors = ["#e74c3c" if l == "Driver" else "#95a5a6" for l in top20["Label"]]
bars = plt.barh(top20["Gene"][::-1], top20["Mutation_Frequency"][::-1], color=colors[::-1])
plt.xlabel("% of Patients Mutated", fontsize=13)
plt.title("Top 20 Most Mutated Genes in Pan-Glioma\n(Red = Statistically Significant Driver)", fontsize=14)
plt.axvline(x=5, color="black", linestyle="--", linewidth=1, label="5% background rate")
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig("graph1_top20_genes.png", bbox_inches="tight")
plt.show()
print("Graph 1 saved")

# ── Graph 2: Volcano Plot ────────────────────────────────────────────────────
plt.figure(figsize=(10, 7))
passengers = results_df[results_df["Label"] == "Passenger"]
drivers = results_df[results_df["Label"] == "Driver"]

# Plot passengers
plt.scatter(
    passengers["Mutation_Frequency"],
    -np.log10(passengers["P_Value"] + 1e-300),
    color="#95a5a6", alpha=0.4, s=15, label="Passenger"
)

# Plot drivers
plt.scatter(
    drivers["Mutation_Frequency"],
    -np.log10(drivers["P_Value"] + 1e-300),
    color="#e74c3c", s=60, label="Driver", zorder=5
)

# Label driver genes
for _, row in drivers.iterrows():
    plt.annotate(
        row["Gene"],
        xy=(row["Mutation_Frequency"], -np.log10(row["P_Value"] + 1e-300)),
        xytext=(8, 2), textcoords="offset points", fontsize=9, color="#c0392b"
    )

plt.axhline(y=-np.log10(bonferroni_threshold), color="black",
            linestyle="--", linewidth=1, label="Bonferroni threshold")
plt.xlabel("Mutation Frequency (% of Patients)", fontsize=13)
plt.ylabel("-log10(P-Value)", fontsize=13)
plt.title("Volcano Plot: Driver vs Passenger Mutation Genes\nin Pan-Glioma (TCGA LGG + GBM)", fontsize=14)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig("graph2_volcano_plot.png", bbox_inches="tight")
plt.show()
print("Graph 2 saved")

# ── Graph 3: Confusion Matrix Heatmap ───────────────────────────────────────
cm = confusion_matrix(y_test, y_pred, labels=["Driver", "Passenger"])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
            xticklabels=["Driver", "Passenger"],
            yticklabels=["Driver", "Passenger"],
            linewidths=0.5)
plt.xlabel("Predicted Label", fontsize=13)
plt.ylabel("True Label", fontsize=13)
plt.title("Confusion Matrix — Random Forest Model\nDriver vs Passenger Gene Classification", fontsize=13)
plt.tight_layout()
plt.savefig("graph3_confusion_matrix.png", bbox_inches="tight")
plt.show()
print("Graph 3 saved")

# ── Graph 4: Feature Importance ──────────────────────────────────────────────
plt.figure(figsize=(8, 5))
sns.barplot(data=importance_df, x="Importance", y="Feature", palette="Reds_r")
plt.xlabel("Importance Score", fontsize=13)
plt.title("Random Forest Feature Importance\nWhat Distinguishes Driver from Passenger Genes", fontsize=13)
plt.tight_layout()
plt.savefig("graph4_feature_importance.png", bbox_inches="tight")
plt.show()
print("Graph 4 saved")

# ── Graph 5: Mutation Type Breakdown ─────────────────────────────────────────
mut_counts = df["Variant_Classification"].value_counts()
plt.figure(figsize=(10, 6))
sns.barplot(x=mut_counts.values, y=mut_counts.index, palette="Blues_r")
plt.xlabel("Number of Mutations", fontsize=13)
plt.title("Mutation Type Distribution Across Pan-Glioma Dataset\n(TCGA LGG + GBM, n=902 patients)", fontsize=13)
plt.tight_layout()
plt.savefig("graph5_mutation_types.png", bbox_inches="tight")
plt.show()
print("Graph 5 saved")

print("\nAll graphs saved to your project folder!")