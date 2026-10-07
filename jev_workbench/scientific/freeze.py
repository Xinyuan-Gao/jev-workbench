"""Immutable local freeze record; not an externally registered preregistration."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

REQUIRED_GATES=('provenance_license','human_gold','independent_test_units','duplicate_audit',
                'local_validation_selection','remote_dev_compatibility','protocol_tests',
                'fixed_analysis','model_identity','executable_budget')


def _canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)


def _artifact(root,path):
    relative=Path(path)
    absolute=(root/relative).resolve()
    if relative.is_absolute() or not absolute.is_relative_to(root):
        raise ValueError('Freeze artifacts must be relative paths within scientific root')
    if any(part.startswith('.env') or part.casefold() in {'credentials','secrets'} for part in relative.parts):
        raise ValueError('Credential files cannot be freeze artifacts')
    if not absolute.is_file(): raise ValueError('Missing freeze artifact: '+str(path))
    digest=hashlib.sha256()
    with absolute.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return {'path':str(relative),'sha256':digest.hexdigest(),'bytes':absolute.stat().st_size}


def freeze_bundle(root,artifact_paths,gates,metadata):
    root=Path(root).resolve();target=root/'frozen.json'
    if target.exists(): raise FileExistsError('Create a new experiment directory instead of overwriting its freeze')
    failed=[name for name in REQUIRED_GATES if not isinstance(gates.get(name),dict)
            or gates[name].get('passed') is not True or not gates[name].get('evidence')]
    if failed: raise ValueError('Unpassed scientific gates: '+', '.join(failed))
    if not artifact_paths or len(set(map(str,artifact_paths)))!=len(artifact_paths):
        raise ValueError('Nonempty unique artifact paths required')
    artifacts=[_artifact(root,path) for path in sorted(map(str,artifact_paths))]
    bundle={'schema_version':1,'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
            'registration':'local immutable record; no external timestamp certification',
            'artifacts':artifacts,'gates':gates,'metadata':metadata}
    bundle['bundle_sha256']=hashlib.sha256(_canonical(bundle).encode()).hexdigest()
    # Exclusive create: accidental rerun cannot replace the protocol after results.
    with target.open('x') as stream: stream.write(json.dumps(bundle,ensure_ascii=False,indent=2,allow_nan=False))
    return bundle


def verify_freeze(root,bundle):
    root=Path(root).resolve();copy={k:v for k,v in bundle.items() if k!='bundle_sha256'}
    valid_hash=hashlib.sha256(_canonical(copy).encode()).hexdigest()==bundle.get('bundle_sha256')
    changed=[]
    for artifact in bundle.get('artifacts',[]):
        try: current=_artifact(root,artifact['path'])
        except ValueError: changed.append(artifact['path']);continue
        if current['sha256']!=artifact['sha256']:changed.append(artifact['path'])
    return {'valid':valid_hash and not changed,'bundle_hash_valid':valid_hash,'changed':changed}
