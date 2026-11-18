import argparse
import csv
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('data_root', type=Path)
a = p.parse_args()
folder = Path(__file__).resolve().parents[1] / 'splits'
parts = {}
missing = []
for split in ('train', 'val', 'test'):
    with (folder / f'{split}.csv').open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    assert rows and {r['label'] for r in rows} == {'1', '2', '3'}
    assert all(r['split'] == split for r in rows)
    assert len({r['relative_path'] for r in rows}) == len(rows)
    parts[split] = rows
    missing.extend(r['relative_path'] for r in rows if not (a.data_root / r['relative_path']).is_file())
    print(f'{split}: {len(rows)} images, {len({r["split_group"] for r in rows})} groups')
for key in ('relative_path', 'patient_id', 'split_group'):
    sets = [{r[key] for r in rows} for rows in parts.values()]
    assert all(not sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)), key
assert sum(map(len, parts.values())) == 3064
print('PASS: split integrity and class coverage')
if missing:
    print(f'Missing files: {len(missing)}. First 10:')
    print('\n'.join(missing[:10]))
    raise SystemExit(1)
print('PASS: all 3064 image paths exist. Image contents and near-duplicates were not checked.')
