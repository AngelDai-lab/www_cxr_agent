"""Build trusted study-level U-Zeros labels; compare a trusted local legacy pickle.

Keys in the output are decimal study_id strings. No training files are changed.
"""
import argparse
import collections
import csv
import gzip
import hashlib
import json
import math
import pickle
import re
from pathlib import Path

LABELS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
          'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity',
          'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia',
          'Pneumothorax', 'Support Devices']


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def study_key(value):
    value = str(value).strip()
    if not re.fullmatch(r'[0-9]+', value):
        raise ValueError(f'Invalid study_id: {value!r}')
    return str(int(value))


def uzeros(value):
    text = str(value).strip()
    if not text or text.lower() == 'nan':
        return 0
    number = float(text)
    if math.isnan(number):
        return 0
    if number not in (-1, 0, 1):
        raise ValueError(f'Unexpected label value: {value!r}')
    return int(number == 1)


def read_official(path, expected_count=None):
    labels = {}
    subjects = {}
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)):
            raise ValueError('Duplicate CSV column names')
        if set(fields) != set(['subject_id', 'study_id'] + LABELS):
            raise ValueError(f'CSV column names mismatch: {fields}; expected subject_id, study_id, {LABELS}')
        for line, row in enumerate(reader, 2):
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f'Malformed CSV row {line}')
            sid = study_key(row['study_id'])
            if sid in labels:
                raise ValueError(f'Duplicate study_id at row {line}: {sid}')
            subjects[sid] = study_key(row['subject_id'])
            try:
                labels[sid] = [uzeros(row[name]) for name in LABELS]
            except ValueError as exc:
                raise ValueError(f'Row {line}, study {sid}: {exc}') from exc
    if not labels:
        raise ValueError('Empty official CSV')
    if expected_count is not None and len(labels) != expected_count:
        raise ValueError(f'Expected {expected_count} studies, found {len(labels)}')
    return labels, subjects


def compare(labels, subjects, old_path, annotation_path, old_order):
    # Only use a pickle from a trusted source: pickle can execute code.
    with Path(old_path).open('rb') as stream:
        old = pickle.load(stream)
    if not isinstance(old, dict):
        raise ValueError('Legacy pickle must be a dictionary')
    order = json.loads(Path(old_order).read_text())
    order = order['labels'] if isinstance(order, dict) else order
    if len(order) != 14 or len(set(order)) != 14 or set(order) != set(LABELS):
        raise ValueError('Legacy order must name exactly the official 14 classes')
    permutation = [order.index(name) for name in LABELS]
    mapping = {}
    annotation = json.loads(Path(annotation_path).read_text())
    for records in annotation.values():
        for record in records:
            sid = study_key(record['study_id'])
            if sid in subjects and record.get('subject_id') is not None:
                if study_key(record['subject_id']) != subjects[sid]:
                    raise ValueError(f'Annotation subject mismatch for study {sid}')
            for key in [str(record['id'])] + [Path(p).stem for p in record.get('image_path', [])]:
                if key in mapping and mapping[key] != sid:
                    raise ValueError(f'Ambiguous image-to-study mapping: {key}')
                mapping[key] = sid
            for p in record.get('image_path', []):
                paths = [part[1:] for part in Path(p).parts if re.fullmatch(r's[0-9]+', part)]
                if paths and any(study_key(v) != sid for v in paths):
                    raise ValueError(f'Image path study mismatch: {p}')
    counts = collections.Counter()
    differences = [0] * 14
    examples = []
    study_vectors = collections.defaultdict(set)
    for raw_key, vector in old.items():
        key = str(raw_key)
        sid = mapping.get(key)
        if sid is None:
            counts['unmapped_old_keys'] += 1
            continue
        if sid not in labels:
            counts['mapped_studies_missing_official'] += 1
            continue
        values = list(vector)
        if len(values) != 14 or any(v not in (0, 1) for v in values):
            raise ValueError(f'Invalid legacy binary vector: {key}')
        values = [int(values[i]) for i in permutation]
        changes = [i for i in range(14) if values[i] != labels[sid][i]]
        counts['compared_images'] += 1
        counts['different_images'] += bool(changes)
        study_vectors[sid].add(tuple(values))
        for i in changes:
            differences[i] += 1
        if changes and len(examples) < 10:
            examples.append({'old_key': key, 'study_id': sid, 'old': values,
                             'official': labels[sid], 'different_labels': [LABELS[i] for i in changes]})
    return {'old_sha256': digest(old_path), 'annotation_sha256': digest(annotation_path),
            'old_order_assumption': order, 'old_order_file_sha256': digest(old_order),
            'counts': dict(counts), 'compared_studies': len(study_vectors),
            'within_study_inconsistent_old_labels': sum(len(v) > 1 for v in study_vectors.values()),
            'image_disagreement_rate': counts['different_images'] / counts['compared_images'] if counts['compared_images'] else None,
            'different_images_per_label': dict(zip(LABELS, differences)), 'examples': examples,
            'note': 'Comparison is image-weighted and conditional on the declared legacy column order; unmapped keys are not mismatches.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--expected-count', type=int)
    parser.add_argument('--old-pickle', type=Path)
    parser.add_argument('--annotation', type=Path)
    parser.add_argument('--old-order-json', type=Path)
    args = parser.parse_args()
    if any((args.old_pickle, args.annotation, args.old_order_json)) and not all((args.old_pickle, args.annotation, args.old_order_json)):
        parser.error('Comparison requires --old-pickle, --annotation and --old-order-json together')
    metadata_path = args.out.with_suffix('.metadata.json')
    if args.out.exists() or metadata_path.exists():
        parser.error('Output already exists; choose a new path to preserve existing artifacts')
    labels, subjects = read_official(args.csv, args.expected_count)
    metadata = {'source': str(args.csv.resolve()), 'source_sha256': digest(args.csv),
                'key_type': 'study_id_string', 'labels': LABELS, 'policy': 'U-Zeros: -1/NaN/blank/0 -> 0; 1 -> 1',
                'n_studies': len(labels), 'expected_count': args.expected_count,
                'positive_counts': dict(zip(LABELS, map(sum, zip(*labels.values()))))}
    if args.old_pickle:
        metadata['legacy_comparison'] = compare(labels, subjects, args.old_pickle, args.annotation, args.old_order_json)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('xb') as stream:
        pickle.dump(labels, stream, protocol=pickle.HIGHEST_PROTOCOL)
    with args.out.open('rb') as stream:
        assert pickle.load(stream) == labels, 'Round-trip verification failed'
    metadata['sha256'] = digest(args.out)
    with metadata_path.open('x') as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False)
    print(json.dumps({'out': str(args.out), 'metadata': str(metadata_path), 'n_studies': len(labels),
                      'comparison': metadata.get('legacy_comparison', {}).get('counts')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
