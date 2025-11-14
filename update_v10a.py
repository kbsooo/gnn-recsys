#!/usr/bin/env python3
"""V9c → V10a 변환: embedding_dim 64 → 128"""

import json
import sys

notebook_path = 'notebooks/gnn_recsys_v10a.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 수정
for i, cell in enumerate(nb['cells']):
    source = ''.join(cell.get('source', []))

    # Cell 0: 제목
    if i == 0 and cell['cell_type'] == 'markdown':
        cell['source'] = [
            "# GNN 기반 영화 추천 시스템\n",
            "# V10A: Embedding Capacity Increase (64 → 128)\n",
            "# Step 2 - Model capacity breakthrough"
        ]

    # Cell 2: CONFIG
    elif "'embedding_dim': 64" in source:
        new_source = []
        for line in cell['source']:
            if "'embedding_dim': 64" in line:
                new_source.append("    'embedding_dim': 128,\n")
            elif "V9C 설정 완료" in line:
                new_source.append('print("V10A 설정 완료! (Embedding Capacity Increase)")\n')
            elif "⭐ V8a: neg_ratio=4 → V9c: neg_ratio=6" in line:
                new_source.append('print("\\n⭐ V9c: embedding_dim=64 → V10a: embedding_dim=128")\n')
            elif "Negative ratio: {CONFIG['neg_ratio']}" in line:
                new_source.append(line)
                new_source.append('print(f"  Embedding dim: {CONFIG[\'embedding_dim\']} (V9c: 64)")\n')
            elif "목표: Recall@10 > 17%" in line:
                new_source.append('print("\\n🎯 목표: Recall@10 > 18% (V9c의 15.78% 대비 breakthrough!)")\n')
            else:
                new_source.append(line)
        cell['source'] = new_source

    # Training cell
    elif "Training 시작 (V9C" in source:
        cell['source'] = [line.replace('V9C', 'V10A').replace('v9c', 'v10a') for line in cell['source']]

    # 시각화
    elif "training_curves_v9c.png" in source:
        cell['source'] = [line.replace('v9c', 'v10a').replace('V9C', 'V10A') for line in cell['source']]

    # Test 평가
    elif "lightgcn_v9c_best.pth" in source:
        cell['source'] = [line.replace('v9c', 'v10a').replace('V9C', 'V10A') for line in cell['source']]

    # 비교 cell
    elif "V9C vs V8a 결과 비교" in source:
        cell['source'] = [line.replace('V9C', 'V10A').replace('v9c', 'v10a') for line in cell['source']]

# 저장
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"✓ V10a created: embedding_dim=128")
