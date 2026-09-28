"""Подготовить локальные веса E5 или проверить уже имеющиеся файлы.

В полном ZIP веса уже есть. После git clone они загружаются отдельно, так как
один файл весов превышает лимит обычного файла GitHub. Предсказание работает
офлайн: сетевые запросы есть только в этой явно запускаемой подготовке.
"""
import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare(source=None, verify_only=False, model_dir=None):
    model_dir = Path(model_dir) if model_dir is not None else ROOT / 'models'
    manifest = json.loads((model_dir / 'model_manifest.json').read_text())
    target = model_dir / 'e5-small'
    for entry in manifest['files']:
        path = target / entry['path']
        if path.is_file() and sha256(path) == entry['sha256']:
            print('OK', entry['path'])
            continue
        if verify_only:
            raise RuntimeError(f"Не найден или изменён файл {entry['path']}. Запустите setup_assets.py.")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + '.part')
        if source:
            shutil.copyfile(Path(source) / entry['path'], temporary)
        else:
            url = f"https://huggingface.co/{manifest['model']}/resolve/{manifest['revision']}/{entry['path']}"
            request = urllib.request.Request(url, headers={'User-Agent': 'Avito-DS-local-reproduction'})
            print('Загрузка', entry['path'], flush=True)
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open('wb') as file:
                shutil.copyfileobj(response, file)
        if temporary.stat().st_size != entry['bytes'] or sha256(temporary) != entry['sha256']:
            raise RuntimeError(f"Не совпала контрольная сумма {entry['path']}; файл не принят.")
        temporary.replace(path)
    print('Все файлы E5 готовы; их SHA-256 совпадают с manifest.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, help='Локальный каталог с файлами E5: копирование без сети.')
    parser.add_argument('--verify-only', action='store_true', help='Только проверка, без копирования и скачивания.')
    parser.add_argument('--model-dir', type=Path, default=ROOT / 'models')
    args = parser.parse_args()
    prepare(args.source, args.verify_only, args.model_dir)
