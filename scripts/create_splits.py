"""Create deterministic class-stratified group splits using only Python's standard library."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--out', type=Path, default=Path('reports/cheng/splits_seed42'))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    with args.manifest.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    required = {'relative_path', 'patient_id', 'split_group', 'label', 'class_name'}
    if not fields or not required.issubset(fields) or not rows:
        raise ValueError('Manifest is empty or missing required columns')
    if len({r['relative_path'] for r in rows}) != len(rows):
        raise ValueError('Duplicate file paths in manifest')
    groups = defaultdict(list)
    patient_groups = defaultdict(set)
    for r in rows:
        if any(not r[k].strip() for k in required):
            raise ValueError('Empty required value')
        if r.get('grouping_status', '').startswith('unresolved'):
            raise ValueError('Resolve grouping decisions before splitting')
        groups[r['split_group']].append(r)
        patient_groups[r['patient_id']].add(r['split_group'])
    if any(len(v) != 1 for v in patient_groups.values()):
        raise ValueError('Patient ID assigned to multiple groups')
    classes = defaultdict(list)
    for group, records in groups.items():
        labels = {(r['label'], r['class_name']) for r in records}
        if len(labels) != 1:
            raise ValueError(f'Conflicting labels in group {group}')
        classes[next(iter(labels))].append(group)
    assignments = {}
    for label, group_ids in sorted(classes.items()):
        # Stable hash order makes assignment independent of input row order.
        ordered = sorted(group_ids, key=lambda g: (
            hashlib.sha256(f'{args.seed}:{g}'.encode()).hexdigest(), g))
        n = len(ordered)
        holdout = max(1, int(n * 0.15 + 0.5))
        if n - 2 * holdout < 1:
            raise ValueError(f'Too few groups for class {label}')
        for i, group in enumerate(ordered):
            assignments[group] = ('test' if i < holdout else
                                  'val' if i < 2 * holdout else 'train')
    output_rows = [dict(r, split=assignments[r['split_group']])
                   for r in sorted(rows, key=lambda r: r['relative_path'])]
    partitions = {s: [r for r in output_rows if r['split'] == s]
                  for s in ('train', 'val', 'test')}
    # Explicitly check disjoint groups, original patient IDs and files.
    for key in ('split_group', 'patient_id', 'relative_path'):
        sets = [{r[key] for r in partitions[s]} for s in partitions]
        if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError(f'Overlap detected: {key}')
    assert sum(map(len, partitions.values())) == len(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    columns = [f for f in fields if f != 'split'] + ['split']
    def save(name, records, columns):
        with (args.out / name).open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=columns)
            w.writeheader()
            w.writerows(records)
    for split, records in partitions.items():
        save(f'{split}.csv', records, columns)
    save('manifest_with_splits.csv', output_rows, columns)
    summary = []
    for split, records in partitions.items():
        for label, name in sorted(classes):
            subset = [r for r in records if r['label'] == label]
            summary.append(dict(split=split, class_name=name,
                                images=len(subset),
                                groups=len({r['split_group'] for r in subset})))
    save('split_summary.csv', summary, ['split', 'class_name', 'images', 'groups'])
    metadata = dict(seed=args.seed, target_ratios=dict(train=.7, val=.15, test=.15),
                    algorithm='SHA256(seed:group) order within class; round 15% groups per holdout',
                    input_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                    images=len(rows), groups=len(groups),
                    original_patient_ids=len(patient_groups),
                    overlap_checks='passed: groups, patient IDs, file paths',
                    near_duplicate_audit='pending; not checked by this script')
    (args.out / 'split_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    for split, records in partitions.items():
        print(f'{split}: {len(records)} images; {len({r["split_group"] for r in records})} groups')
    print('PASS: every image assigned once; no group, patient-ID or file overlap.')
    print(f'Output: {args.out.resolve()}')


if __name__ == '__main__':
    main()
