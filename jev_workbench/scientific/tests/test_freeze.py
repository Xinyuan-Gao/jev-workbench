import importlib
import pytest


def module(): return importlib.import_module('jev_workbench.scientific.freeze')


def gates():
    return {name:{'passed':True,'evidence':'fixture only'} for name in module().REQUIRED_GATES}


def test_freeze_refuses_unchecked_or_failed_gates(tmp_path):
    source=tmp_path/'protocol.md';source.write_text('fixed protocol')
    g=gates();g['human_gold']['passed']=False
    with pytest.raises(ValueError): module().freeze_bundle(tmp_path,['protocol.md'],g,{})
    g.pop('human_gold')
    with pytest.raises(ValueError): module().freeze_bundle(tmp_path,['protocol.md'],g,{})
    assert not (tmp_path/'frozen.json').exists()


def test_freeze_hashes_actual_files_and_detects_changed_input(tmp_path):
    source=tmp_path/'manifest.json';source.write_text('test data')
    frozen=module().freeze_bundle(tmp_path,['manifest.json'],gates(),{'scope':'fixture'})
    assert module().verify_freeze(tmp_path,frozen)['valid']
    source.write_text('changed test data')
    result=module().verify_freeze(tmp_path,frozen)
    assert not result['valid'] and result['changed']==['manifest.json']


def test_freeze_rejects_credentials_and_outside_paths(tmp_path):
    secret=tmp_path/'.env.local';secret.write_text('fake credential')
    with pytest.raises(ValueError): module().freeze_bundle(tmp_path,['.env.local'],gates(),{})
    with pytest.raises(ValueError): module().freeze_bundle(tmp_path,['../outside'],gates(),{})


def test_existing_freeze_cannot_be_replaced(tmp_path):
    (tmp_path/'protocol.md').write_text('fixed')
    module().freeze_bundle(tmp_path,['protocol.md'],gates(),{})
    with pytest.raises(FileExistsError): module().freeze_bundle(tmp_path,['protocol.md'],gates(),{})
