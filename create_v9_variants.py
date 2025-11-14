#!/usr/bin/env python3
"""
V8a를 기반으로 V9a-d 노트북 생성
temperature와 neg_ratio만 변경
"""

import json
import sys

def create_v9_variant(input_file, output_file, variant_name, temperature, neg_ratio):
    """V8a 노트북을 읽어서 V9 variant 생성"""

    # 노트북 로드
    with open(input_file, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    # Variant 정보
    variant_info = {
        'v9a': ('sharper, more aggressive', 'V8a: temp=0.2 → V9a: temp=0.1 (더 sharp)'),
        'v9b': ('softer, more stable', 'V8a: temp=0.2 → V9b: temp=0.3 (더 soft)'),
        'v9c': ('more negatives', 'V8a: neg_ratio=4 → V9c: neg_ratio=6'),
        'v9d': ('soft + more negatives (expected best)', 'V8a: temp=0.2, neg_ratio=4 → V9d: temp=0.3, neg_ratio=6'),
    }

    desc, change_desc = variant_info[variant_name]

    # Cell별 수정
    for i, cell in enumerate(notebook['cells']):
        source = ''.join(cell.get('source', []))

        # Cell 0: 제목 변경
        if i == 0 and cell['cell_type'] == 'markdown':
            cell['source'] = [
                f"# GNN 기반 영화 추천 시스템\n",
                f"# {variant_name.upper()}: Temperature & Negative Ratio Tuning\n",
                f"# Step 1 실험 - {desc}"
            ]

        # Cell 2: CONFIG 수정
        elif "'temperature':" in source and "'neg_ratio':" in source:
            # CONFIG 셀 찾음
            new_source = []
            in_config = False

            for line in cell['source']:
                # Temperature 라인
                if "'temperature':" in line:
                    new_source.append(f"    'temperature': {temperature},  # V9 tuning: {temperature}\n")
                # Neg ratio 라인
                elif "'neg_ratio':" in line:
                    new_source.append(f"    'neg_ratio': {neg_ratio},\n")
                # 출력 메시지 수정
                elif 'V8a 설정 완료' in line:
                    new_source.append(f'print("{variant_name.upper()} 설정 완료! (Temperature & Neg Ratio Tuning)")\n')
                elif 'V6 → V8a 변경사항' in line:
                    new_source.append(f'print("\\n⭐ {change_desc}")\n')
                elif 'Temperature:' in line and 'score diversity' in line:
                    new_source.append(f'print(f"  Temperature: {{CONFIG[\'temperature\']}} (V8a: 0.2)")\n')
                elif 'Negative ratio:' in line and 'CONFIG' not in line:
                    new_source.append(f'print(f"  Negative ratio: {{CONFIG[\'neg_ratio\']}} (V8a: 4)")\n')
                elif '목표: Recall@10 > 9%' in line:
                    new_source.append(f'print("\\n🎯 목표: Recall@10 > 17% (V8a의 15.41% 대비 개선!)")\n')
                else:
                    new_source.append(line)

            cell['source'] = new_source

        # Cell 14: Training - 모델 저장 경로
        elif 'lightgcn_v8a_best.pth' in source and 'Training 시작' in source:
            cell['source'] = [line.replace('v8a', variant_name).replace('V8a', variant_name.upper()) for line in cell['source']]

        # Cell 15: 시각화 - 이미지 저장 경로
        elif 'training_curves_v8a.png' in source:
            cell['source'] = [line.replace('v8a', variant_name).replace('V8a', variant_name.upper()) for line in cell['source']]

        # Cell 17: Test 평가 - 모델 로드 경로
        elif 'Test Set 평가' in source and 'lightgcn_v8a_best.pth' in source:
            cell['source'] = [line.replace('v8a', variant_name).replace('V8a', variant_name.upper()) for line in cell['source']]

        # Cell 19: 비교 - V8a vs V9x
        elif 'V8a vs V6 결과 비교' in source:
            new_source = []
            for line in cell['source']:
                if 'V8a vs V6 결과 비교' in line:
                    new_source.append(f'print("{variant_name.upper()} vs V8a 결과 비교")\n')
                elif "V6:  BPR Loss" in line:
                    new_source.append(f'print(f"  V8a: temp=0.2, neg_ratio=4")\n')
                elif "V8a: InfoNCE Loss" in line:
                    new_source.append(f'print(f"  {variant_name}: temp={{CONFIG[\'temperature\']}}, neg_ratio={{CONFIG[\'neg_ratio\']}}")\n')
                elif 'V6:  {v6_results' in line:
                    new_source.append(f'print(f"  V8a: {{v6_results[10][\'recall@10\']:.4f}} (15.41%)")\n')
                elif 'V8a: {v8a_results' in line:
                    new_source.append(f'print(f"  {variant_name}: {{v8a_results[10][\'recall@10\']:.4f}} ({{v8a_results[10][\'recall@10\']*100:.2f}}%)")\n')
                elif '# V6 결과' in line:
                    new_source.append(f'# V8a 결과 (baseline)\n')
                elif "10: {'precision@10': 0.1633" in line:
                    new_source.append(f"    10: {{'precision@10': 0.2726, 'recall@10': 0.1541, 'ndcg@10': 0.3093}}\n")
                elif 'V6=' in line and 'V8a=' in line:
                    line_modified = line.replace('V6=', 'V8a=').replace('V8a=', f'{variant_name}=')
                    new_source.append(line_modified)
                else:
                    new_source.append(line)

            cell['source'] = new_source

    # 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

    print(f"✓ Created {output_file}")

# 메인
if __name__ == '__main__':
    base_dir = 'notebooks'

    variants = [
        ('v9a', 0.1, 4),
        ('v9b', 0.3, 4),
        ('v9c', 0.2, 6),
        ('v9d', 0.3, 6),
    ]

    for variant_name, temperature, neg_ratio in variants:
        input_file = f'{base_dir}/gnn_recsys_v8a.ipynb'
        output_file = f'{base_dir}/gnn_recsys_{variant_name}.ipynb'
        create_v9_variant(input_file, output_file, variant_name, temperature, neg_ratio)

    print("\n✅ All V9 variants created!")
    print("\nNext: Run all 4 notebooks in parallel or sequentially")
