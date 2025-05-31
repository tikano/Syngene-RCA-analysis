#!/usr/bin/env python
# coding: utf-8

# # SynGene-Bench RCA Generator — Full Neo4j Format
# ---
# This notebook generates synthetic data for SynGene RCA benchmarking.
# Output includes 6 node tables and 6 edge tables fully aligned with Neo4j ingestion pipeline.
# ---

# In[1]:



import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
import os


# In[2]:



# CONFIGURABLE PARAMETERS
NUM_BATCHES = 10000
NUM_EQUIPMENT = 300
NUM_OPERATORS = 200
NUM_MATERIALS = 4000
NUM_FAULTS = 100
CASCADE_DEPTH = 3
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

OUTPUT_DIR = './syngene_full_output/'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# In[3]:



equipment = pd.DataFrame({
    'EquipmentID': [f"EQUIP_{i+1}" for i in range(NUM_EQUIPMENT)]
})
equipment.to_csv(f"{OUTPUT_DIR}/equipment.csv", index=False)

materials = pd.DataFrame({
    'MaterialID': [f"MAT_{i+1}" for i in range(NUM_MATERIALS)]
})
materials.to_csv(f"{OUTPUT_DIR}/materials.csv", index=False)

operators = pd.DataFrame({
    'OperatorID': [f"OP_{i+1}" for i in range(NUM_OPERATORS)]
})
operators.to_csv(f"{OUTPUT_DIR}/operators.csv", index=False)


# In[4]:



batches = []
processed_on = []
used_material = []
cleaned_by = []
precedes = []

start_date = datetime(2023, 1, 1)
for i in range(NUM_BATCHES):
    batch_id = f"BATCH_{i+1}"
    equip = f"EQUIP_{random.randint(1, NUM_EQUIPMENT)}"
    mat = f"MAT_{random.randint(1, NUM_MATERIALS)}"
    op = f"OP_{random.randint(1, NUM_OPERATORS)}"
    start = start_date + timedelta(hours=i*2)
    end = start + timedelta(hours=2)
    parent = f"BATCH_{i}" if i > 0 and random.random() < 0.5 else None

    batches.append([batch_id, start.isoformat(), end.isoformat()])
    processed_on.append([batch_id, equip])
    used_material.append([batch_id, mat])
    cleaned_by.append([equip, op])
    if parent:
        precedes.append([parent, batch_id])

df_batches = pd.DataFrame(batches, columns=["BatchID", "StartTime", "EndTime"])
df_batches.to_csv(f"{OUTPUT_DIR}/batches.csv", index=False)
pd.DataFrame(processed_on, columns=["BatchID", "EquipmentID"]).to_csv(f"{OUTPUT_DIR}/processed_on.csv", index=False)
pd.DataFrame(used_material, columns=["BatchID", "MaterialID"]).to_csv(f"{OUTPUT_DIR}/used_material.csv", index=False)
pd.DataFrame(cleaned_by, columns=["EquipmentID", "OperatorID"]).to_csv(f"{OUTPUT_DIR}/cleaned_by.csv", index=False)
pd.DataFrame(precedes, columns=["ParentBatchID", "ChildBatchID"]).to_csv(f"{OUTPUT_DIR}/precedes.csv", index=False)


# In[5]:



fault_sources = df_batches.sample(NUM_FAULTS, random_state=SEED)['BatchID'].tolist()
deviations = []
contaminated_batches = set()

for dev_id, seed_batch in enumerate(fault_sources, start=1):
    queue = [(seed_batch, 0)]
    while queue:
        current_batch, depth = queue.pop(0)
        contaminated_batches.add(current_batch)
        if depth < CASCADE_DEPTH:
            children = df_batches[df_batches['BatchID'].isin(df_batches[df_batches['BatchID'] == current_batch]['BatchID'])]['BatchID'].tolist()
            children += pd.read_csv(f"{OUTPUT_DIR}/precedes.csv")[pd.read_csv(f"{OUTPUT_DIR}/precedes.csv")['ParentBatchID'] == current_batch]['ChildBatchID'].tolist()
            for child in children:
                queue.append((child, depth+1))
    deviations.append([f"DEV_{dev_id}", seed_batch, "ContaminationCascade"])

df_devs = pd.DataFrame(deviations, columns=["DeviationID", "BatchID", "Type"])
df_devs.to_csv(f"{OUTPUT_DIR}/deviations.csv", index=False)
pd.DataFrame({"ContaminatedBatch": list(contaminated_batches)}).to_csv(f"{OUTPUT_DIR}/ground_truth_rca.csv", index=False)

pd.DataFrame([[row.BatchID, row.DeviationID] for idx, row in df_devs.iterrows()], columns=["BatchID", "DeviationID"]).to_csv(f"{OUTPUT_DIR}/has_deviation.csv", index=False)


# In[6]:



labtests = []
has_test = []

for batch in df_batches['BatchID']:
    for _ in range(3):
        test_id = f"TEST_{len(labtests)+1}"
        test_type = random.choice(["Microbial", "Potency", "Endotoxin", "pH"])
        result = random.choice(["PASS", "FAIL"])
        labtests.append([test_id, test_type, result])
        has_test.append([batch, test_id])

df_tests = pd.DataFrame(labtests, columns=["TestID", "TestType", "Result"])
df_tests.to_csv(f"{OUTPUT_DIR}/labtests.csv", index=False)
pd.DataFrame(has_test, columns=["BatchID", "TestID"]).to_csv(f"{OUTPUT_DIR}/has_test.csv", index=False)


# ✅ Full RCA SynGene dataset generated with 12 CSV files:
# 
# - Node files: `batches.csv`, `equipment.csv`, `materials.csv`, `operators.csv`, `labtests.csv`, `deviations.csv`
# - Edge files: `processed_on.csv`, `used_material.csv`, `cleaned_by.csv`, `precedes.csv`, `has_test.csv`, `has_deviation.csv`
# - RCA ground truth: `ground_truth_rca.csv`

# In[ ]:




