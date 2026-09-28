"""Заново рассчитать отправленный answer.csv и проверить его точное совпадение.

Файл из корня проекта не копируется в ответ и не используется как признак.
Сначала выполняется обычный поиск по исходным Parquet, затем проверяется SHA.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
EXPECTED_SHA256 = 'ef4ba4b73e9cbb7821d4b7c2a1899b97f8c962adde06a81a70f81dd51c9910ea'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--model-dir', type=Path, default=ROOT / 'models')
    parser.add_argument('--output-dir', type=Path, default=Path('results/reproduced'))
    parser.add_argument('--cache-dir', type=Path, default=Path('cache/predict'))
    parser.add_argument('--device', default='cpu', choices=['cpu', 'mps', 'cuda'])
    args = parser.parse_args()
    for name in ['train.parquet', 'benchmark_items.parquet', 'benchmark_queries.parquet']:
        if not (args.data_dir / name).is_file():
            raise FileNotFoundError(f'В --data-dir отсутствует {name}')
    subprocess.run([sys.executable, str(ROOT / 'setup_assets.py'), '--verify-only',
                    '--model-dir', str(args.model_dir.resolve())], check=True)
    command = [sys.executable, str(ROOT / 'solution_pipeline.py'), 'predict',
               '--data-dir', str(args.data_dir.resolve()), '--model-dir', str(args.model_dir.resolve()),
               '--output-dir', str(args.output_dir.resolve()), '--cache-dir', str(args.cache_dir.resolve()),
               '--device', args.device]
    subprocess.run(command, check=True)
    from validate_submission import validate
    answer = args.output_dir / 'answer.csv'
    report = validate(answer, args.data_dir / 'benchmark_queries.parquet',
                      args.data_dir / 'benchmark_items.parquet', require_50=True)
    report['expected_sha256'] = EXPECTED_SHA256
    report['byte_identical_to_submitted'] = report['sha256'] == EXPECTED_SHA256
    (args.output_dir / 'reproduction_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    if not report['valid']:
        raise RuntimeError('Полученный CSV не прошёл проверку формата; смотрите reproduction_report.json.')
    if not report['byte_identical_to_submitted']:
        raise RuntimeError('CSV рассчитан, но не совпал побайтно с отправленным. Сверьте версии библиотек, устройство и кэш.')
    print('Успех: заново рассчитанный answer.csv побайтно совпадает с отправленным.')


if __name__ == '__main__':
    main()
