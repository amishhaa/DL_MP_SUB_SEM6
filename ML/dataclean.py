import re
import pandas as pd
from collections import defaultdict


df = pd.read_csv("../dataset/HDFS.log.csv")
print(df.head())
print(df.columns)

block_sequences = defaultdict(list)

for _, row in df.iterrows():
    content = row["Content"]
    event_id = row["EventId"]
    
    blk_ids = re.findall(r'(blk_-?\d+)', content)
    
    if not blk_ids:
        continue
    
    for blk in blk_ids:
        block_sequences[blk].append(event_id)

print("num blocks:", len(block_sequences))
print("sample:", list(block_sequences.items())[:1])

event_to_int = {}
current_id = 0

sequences = {}

for blk, seq in block_sequences.items():
    new_seq = []
    
    for event in seq:
        if event not in event_to_int:
            event_to_int[event] = current_id
            current_id += 1
        
        new_seq.append(event_to_int[event])
    
    sequences[blk] = new_seq

print("num events:", len(event_to_int))
print("sample seq:", list(sequences.values())[0][:10])

labels_df = pd.read_csv("../dataset/anomaly_label.csv")

label_map = dict(zip(labels_df["BlockId"], labels_df["Label"]))

def get_label(blk):
    return 1 if label_map.get(blk) == "Anomaly" else 0

window_size = 10

X = []
y = []

for blk, seq in sequences.items():
    if len(seq) < window_size:
        continue
    
    label = get_label(blk)
    
    for i in range(len(seq) - window_size):
        X.append(seq[i:i+window_size])
        y.append(label)

print("samples:", len(X))
print("sequence length:", len(X[0]))
print("labels:", set(y))

import numpy as np

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)

np.save("../dataset/X.npy", X)
np.save("../dataset/y.npy", y)

print("Saved X and y")