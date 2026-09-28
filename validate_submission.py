"""Независимая проверка answer.csv без загрузки поисковой модели.

Пример: python validate_submission.py --answer answer.csv --data-dir data --require-50
Для сохранения отчёта добавьте --report submission_check.json.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

import pyarrow.parquet as pq


def validate(answer_path, query_path, item_path, require_50=False):
    query_ids = pq.read_table(query_path, columns=['query_id']).column('query_id').to_pylist()
    item_ids = pq.read_table(item_path, columns=['item_id']).column('item_id').to_pylist()
    if not all(isinstance(x, str) for x in query_ids + item_ids):
        raise ValueError('Source identifiers must be strings.')
    if len(set(query_ids)) != len(query_ids) or len(set(item_ids)) != len(item_ids):
        raise ValueError('Source identifiers must be unique.')
    expected_queries, allowed_items = set(query_ids), set(item_ids)
    errors = []
    error_count = 0
    seen_queries, lengths = [], []
    row_count = 0
    total_candidates = 0
    raw = Path(answer_path).read_bytes()
    has_bom = raw.startswith(b'\xef\xbb\xbf')
    header = None

    def fail(message):
        nonlocal error_count
        error_count += 1
        if len(errors) < 50:
            errors.append(message)

    if has_bom:
        fail('UTF-8 BOM is not allowed; save plain UTF-8.')
    try:
        with Path(answer_path).open('r', encoding='utf-8', newline='') as file:
            reader = csv.reader(file, delimiter=',', strict=True)
            header = next(reader, None)
            if header != ['query_id', 'answer']:
                fail('Header must be exactly query_id,answer in that order.')
            for row_number, row in enumerate(reader, start=2):
                row_count += 1
                if len(row) != 2:
                    fail(f'Row {row_number}: expected exactly two CSV columns.')
                    continue
                query_id, answer = row
                seen_queries.append(query_id)
                if len(query_id) != 16:
                    fail(f'Row {row_number}: query_id must have length 16.')
                # No stripping/case-folding: the exact original strings matter.
                candidates = answer.split(' ') if answer else []
                lengths.append(len(candidates))
                total_candidates += len(candidates)
                if len(candidates) > 50 or (require_50 and len(candidates) != 50):
                    fail(f'Row {row_number}: invalid candidate count {len(candidates)}.')
                if len(set(candidates)) != len(candidates):
                    fail(f'Row {row_number}: duplicate item_id within answer.')
                if not all(re.fullmatch(r'[0-9a-f]{16}', x) for x in candidates):
                    fail(f'Row {row_number}: invalid item_id format or spacing.')
                if not set(candidates) <= allowed_items:
                    fail(f'Row {row_number}: item_id is absent from the corpus.')
    except (UnicodeError, csv.Error) as error:
        fail(f'Cannot parse strict UTF-8 CSV: {error}')
    actual_queries = set(seen_queries)
    if row_count != len(query_ids):
        fail(f'Expected {len(query_ids)} rows; found {row_count}.')
    if len(seen_queries) != len(actual_queries):
        fail('Duplicate query_id rows.')
    missing = expected_queries - actual_queries
    extra = actual_queries - expected_queries
    if missing:
        fail(f'Missing {len(missing)} query_id values.')
    if extra:
        fail(f'Found {len(extra)} unexpected query_id values, possibly with altered case.')
    return {
        'valid': error_count == 0, 'error_count': error_count, 'errors': errors,
        'encoding': 'utf-8', 'utf8_bom': has_bom, 'delimiter': ',', 'columns': header,
        'rows': row_count, 'expected_queries': len(query_ids),
        'unique_queries': len(actual_queries), 'missing_queries': len(missing),
        'extra_queries': len(extra), 'require_exactly_50': require_50,
        'min_candidates': min(lengths) if lengths else None,
        'max_candidates': max(lengths) if lengths else None,
        'empty_answers': sum(n == 0 for n in lengths), 'total_candidates': total_candidates,
        'sha256': hashlib.sha256(raw).hexdigest(),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--answer', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--queries', type=Path)
    parser.add_argument('--items', type=Path)
    parser.add_argument('--require-50', action='store_true')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    query_path = args.queries or args.data_dir / 'benchmark_queries.parquet'
    item_path = args.items or args.data_dir / 'benchmark_items.parquet'
    result = validate(args.answer, query_path, item_path, args.require_50)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding='utf-8')
    print(text)
    raise SystemExit(0 if result['valid'] else 1)
