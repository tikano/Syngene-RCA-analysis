#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# syn_gene_traversal_demo.py
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from py2neo import Graph, Node, Relationship
import networkx as nx
import matplotlib.pyplot as plt
import os, json, time, openai

# ------------------------------------------------------------------
# 0.  CONFIGURATION
# ------------------------------------------------------------------
GRAPH_URI  = "bolt://localhost:7687"
USER       = "neo4j"
PASSWORD   = "neo4j"     # <-- change if you set a different password
DATA_DIR   = Path("syngene_bench_template")   # folder with the CSVs
SEED_DEV   = "D1"           # deviation we’ll trace from
MAX_DEPTH  = 3              # traversal depth



openai.api_key = os.getenv("OPENAI_API_KEY")  # export your key first
embedder = SentenceTransformer("all-MiniLM-L6-v2")

index  = faiss.read_index("node_index.faiss")
meta   = json.load(open("node_meta.json"))
id2nid = {i: m['nid'] for i, m in enumerate(meta)}
# ------------------------------------------------------------------
# 1. Prompt templates
# ------------------------------------------------------------------
SYSTEM_PROMPT = """You are a pharmaceutical manufacturing data assistant.
Given a deviation ID, produce a Cypher query that returns:
  - affected_batch (batch_id)
  - fault_path     (list of node IDs in traversal order)
Apply these rules:
  • Traverse PROCESSED_ON → CLEANED_BY → PRECEDES edges
  • Stop at depth ≤ 3
  • Include only equipment whose clean_state='dirty' OR
    operators whose training_status='expired'
Return ONLY valid Cypher in a markdown ```cypher block.```"""

NL_EXPLAIN_PROMPT = """Explain in 3–5 sentences, for a GMP auditor,
why each batch was flagged.  Cite equipment IDs or operator IDs as needed.
Use the JSON payload below:
```json
{result_json}
```"""





# ------------------------------------------------------------------
# 1.  CONNECT TO NEO4J
# ------------------------------------------------------------------
graph = Graph(GRAPH_URI, auth=(USER, PASSWORD))
print(f"✅  Connected to Neo4j at {GRAPH_URI}")

# ------------------------------------------------------------------
# 2.  (RE)LOAD DATA
# ------------------------------------------------------------------
def run(q, **kw):
    return graph.run(q, **kw)

print("• Clearing database …")
run("MATCH (n) DETACH DELETE n")

# 2.1  Constraints
constraints = {
    "Batch":"batch_id", "Material":"material_id", "Equipment":"equipment_id",
    "Operator":"operator_id", "LabTest":"test_id", "Deviation":"dev_id"
}
for label, field in constraints.items():
    run(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.{field} IS UNIQUE")

# 2.2  Helper loaders
def load_nodes(csv_file:Path, label:str, id_field:str):
    df = pd.read_csv(csv_file)
    for rec in tqdm(df.to_dict("records"), desc=f"Nodes:{label}"):
        graph.merge(Node(label, **rec), label, id_field)

def load_edges(csv_file:Path,
               src_field:str, src_label:str,
               rel_type:str,
               tgt_field:str, tgt_label:str):
    df = pd.read_csv(csv_file)
    for rec in tqdm(df.to_dict("records"), desc=f"Edges:{rel_type}"):
        s = graph.nodes.match(src_label, **{src_field:rec[src_field]}).first()
        t = graph.nodes.match(tgt_label, **{tgt_field:rec[tgt_field]}).first()
        if s and t:
            graph.merge(Relationship(s, rel_type, t))

# 2.3  Load all CSVs
load_nodes(DATA_DIR/"batches.csv",    "Batch",    "batch_id")
load_nodes(DATA_DIR/"materials.csv",  "Material", "material_id")
load_nodes(DATA_DIR/"equipment.csv",  "Equipment","equipment_id")
load_nodes(DATA_DIR/"operators.csv",  "Operator", "operator_id")
load_nodes(DATA_DIR/"lab_tests.csv",  "LabTest",  "test_id")
load_nodes(DATA_DIR/"deviations.csv", "Deviation","dev_id")

load_edges(DATA_DIR/"used_material.csv", 'batch_id','Batch','USED_MATERIAL','material_id','Material')
load_edges(DATA_DIR/"processed_on.csv",  'batch_id','Batch','PROCESSED_ON',  'equipment_id','Equipment')
load_edges(DATA_DIR/"cleaned_by.csv",    'equipment_id','Equipment','CLEANED_BY','operator_id','Operator')
load_edges(DATA_DIR/"has_test.csv",      'batch_id','Batch','HAS_TEST','test_id','LabTest')
load_edges(DATA_DIR/"has_deviation.csv", 'batch_id','Batch','HAS_DEVIATION','dev_id','Deviation')
load_edges(DATA_DIR/"precedes.csv",      'upstream_batch','Batch','PRECEDES','downstream_batch','Batch')

print("✅  Data loaded into Neo4j")

# ------------------------------------------------------------------
# 3.  RULE-GUIDED TRAVERSAL AGENT
# ------------------------------------------------------------------
def traverse_from_deviation(dev_id:str, max_depth:int=3):
    """
    Returns list of {affected_batch, trace_nodes[]} dictionaries.
    """
    query = f"""
    // Seed: deviation node -> owning batch
    MATCH (d:Deviation {{dev_id:$did}})<-[:HAS_DEVIATION]-(start:Batch)
    // Traversal config (APOC path expander)
    CALL apoc.path.expandConfig(start, {{
        relationshipFilter:'PROCESSED_ON>|CLEANED_BY>|PRECEDES>',
        minLevel:1,
        maxLevel:{max_depth},
        bfs:true,
        uniqueness:'NODE_GLOBAL'
    }}) YIELD path
    WITH DISTINCT path
    WHERE last(nodes(path)) :Batch
    RETURN last(nodes(path)).batch_id AS affected_batch,
           [n IN nodes(path) | labels(n)[0] + ':' +
                 coalesce(n.batch_id,n.equipment_id,n.operator_id,'')] AS trace
    """
    return list(graph.run(query, did=dev_id))

results = traverse_from_deviation(SEED_DEV, MAX_DEPTH)
print(f"\nAffected batches traced from {SEED_DEV}:",
      [r['affected_batch'] for r in results])




def semantic_seed(query_text, k=5):
    qvec = embedder.encode([query_text]).astype('float32')
    D, I = index.search(qvec, k)
    return [id2nid[i] for i in I[0]]

def traverse_subgraph(seed_nids, max_depth=3):
    cypher = f"""
    MATCH (n)
    WHERE id(n) IN $seeds
    CALL apoc.path.subgraphNodes(n,{{
         relationshipFilter:'PROCESSED_ON>|CLEANED_BY>|PRECEDES>',
         maxLevel:{max_depth}, bfs:true}}) YIELD node
    WITH collect(DISTINCT node) AS nodes
    MATCH p=(a)-[*..{max_depth}]-(b) WHERE a IN nodes AND b IN nodes
    RETURN DISTINCT a.batch_id AS batch_id, nodes(p) AS trace
    """
    return graph.run(cypher, seeds=seed_nids).to_data_frame()

def explain_llm(df, user_query):
    explanation = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0125",
        temperature=0.2,
        messages=[
          {"role":"system","content":
           "You are a GMP compliance assistant.  Explain findings based on the KG paths."},
          {"role":"user","content":
           f"User query: {user_query}\nKG result JSON:\n{df.to_json(orient='records', indent=2)}"}
        ]
    ).choices[0].message.content.strip()
    return explanation

# ---------- MAIN -------------
def graphrag_rca(user_query):
    seed_nids = semantic_seed(user_query, k=5)
    df_paths  = traverse_subgraph(seed_nids, max_depth=3)
    affected  = df_paths["batch_id"].dropna().unique().tolist()
    #explanation = explain_llm(df_paths, user_query)
    return affected, df_paths, explanation

results2 = graphrag_rca(SEED_DEV)





# ------------------------------------------------------------------
# 4.  SIMPLE VISUALISATION
# ------------------------------------------------------------------
if results:
    # Visualise first path
    path_nodes = results[0]['trace']
    G = nx.DiGraph()
    for i in range(len(path_nodes)-1):
        G.add_edge(path_nodes[i], path_nodes[i+1])

    plt.figure(figsize=(10, 6))
    nx.draw_networkx(G, node_size=1200, font_size=8, arrows=True)
    plt.title(f"Explanation path for deviation {SEED_DEV}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
else:
    print("No traversal result. Check deviation ID or rules.")



import time
t0 = time.perf_counter()
results = traverse_from_deviation(SEED_DEV, MAX_DEPTH)
runtime = time.perf_counter() - t0

detected = {r['affected_batch'] for r in results}
ground_truth = {'B3', 'B4'}        # <-- replace with true set
precision = len(detected & ground_truth) / len(detected)
recall    = len(detected & ground_truth) / len(ground_truth)
f1        = 2*precision*recall/(precision+recall)

print(f"Runtime: {runtime:.3f} s  |  Precision: {precision:.2f}  Recall: {recall:.2f}  F1: {f1:.2f}")

