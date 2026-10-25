import importlib.util
import json
from pathlib import Path
import sys
import pytest

P=Path(__file__).resolve().parents[1]/'benchmarks'/'run_coverage_benchmarks.py'
spec=importlib.util.spec_from_file_location('coverage_dispatcher',P)
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def entry():
    return {'id':'gaussian','safe_to_execute':True,'argv':['{python}','-m','dev.benchmarks.benchmark_release_refresh','--backends','{backend}','--output','{output}'],'smoke_argv':['{python}','-m','dev.benchmarks.benchmark_release_refresh','--smoke','--output','{output}']}


def test_command_substitution(tmp_path):
    args=m.command(entry(),tmp_path/'a.json','cupy',False)
    assert args[0]==sys.executable
    assert 'cupy' in args
    assert args[-1]==str(tmp_path/'a.json')


def test_unreviewed_script_rejected(tmp_path):
    e=entry(); e['argv'][2]='dev.benchmarks.run_full_benchmark'
    with pytest.raises(ValueError,match='allowlist'):m.command(e,tmp_path/'a','numpy',False)


def test_unsafe_entry_rejected(tmp_path):
    e=entry();e['safe_to_execute']=False
    with pytest.raises(ValueError):m.command(e,tmp_path/'a','numpy',False)


def test_unresolved_template_rejected(tmp_path):
    e=entry();e['argv']+=['{secret}']
    with pytest.raises(ValueError,match='template'):m.command(e,tmp_path/'a','numpy',False)


@pytest.mark.parametrize('payload,expected',[
 ({'status':'complete','rows':[{'status':'failed'}]},'failed'),
 ({'status':'complete','rows':[{'status':'skipped'}]},'partial'),
 ({'validation':{'status':'complete'},'rows':[{'status':'pass'}]},'passed'),
 ({'status':'partial'},'partial'),
 ({'status':'not_applicable'},'failed'),
 ({'status':'complete','cases':[{'runs':[{'status':'failure'}]}]},'failed'),
 ({},'failed'),
])
def test_no_false_success(payload,expected):assert m.disposition(payload)==expected


def test_manifest_duplicates(tmp_path):
    p=tmp_path/'m.json';p.write_text(json.dumps({'entries':[{'id':'a'},{'id':'a'}]}))
    with pytest.raises(ValueError):m.read_manifest(p)


def test_nonzero_exit_is_failure(tmp_path):
    out=tmp_path/'a.json'
    result=m.launch([sys.executable,'-c',f'import pathlib;pathlib.Path({str(out)!r}).write_text(\'{{"status":"pass"}}\');raise SystemExit(3)'],out,10,tmp_path/'a.log')
    assert result['status']=='failed'


def test_missing_output_failure(tmp_path):
    result=m.launch([sys.executable,'-c','pass'],tmp_path/'a.json',10,tmp_path/'a.log')
    assert result['status']=='failed'


def test_timeout_recorded(tmp_path):
    result=m.launch([sys.executable,'-c','import time;time.sleep(30)'],tmp_path/'a.json',0.05,tmp_path/'a.log')
    assert result['reason']=='timeout'


def test_no_overwrite(tmp_path):
    out=tmp_path/'a.json';out.write_text('{}')
    with pytest.raises(ValueError):m.launch([sys.executable,'-c','pass'],out,10,tmp_path/'a.log')


def test_invalid_json_rejected(tmp_path):
    out=tmp_path/'a.json'
    result=m.launch([sys.executable,'-c',f'import pathlib;pathlib.Path({str(out)!r}).write_text(\'{{"value":NaN}}\')'],out,10,tmp_path/'a.log')
    assert result['status']=='failed'


def test_nonzero_partial_cannot_be_laundered(tmp_path):
    out=tmp_path/'partial.json'
    result=m.launch([sys.executable,'-c',f'import pathlib;pathlib.Path({str(out)!r}).write_text(\'{{"status":"partial"}}\');raise SystemExit(3)'],out,10,tmp_path/'partial.log')
    assert result['status']=='failed'


def test_trace_helper_changes_dispatcher_source_fingerprint(tmp_path, monkeypatch):
    scripts = tmp_path / 'dev' / 'benchmarks'
    scripts.mkdir(parents=True)
    dispatcher = scripts / 'run_coverage_benchmarks.py'
    dispatcher.write_text('# test dispatcher\n')
    helper = scripts / 'learning_execution_trace.py'
    helper.write_text('# original observer\n')
    manifest = scripts / 'manifest.json'
    manifest.write_text('{}')
    monkeypatch.setattr(m, 'ROOT', tmp_path)
    monkeypatch.setattr(m, '__file__', str(dispatcher))
    before = m.fingerprint(manifest)
    key = 'dev/benchmarks/learning_execution_trace.py'
    assert key in before['files_sha256']
    helper.write_text('# changed observer contract\n')
    after = m.fingerprint(manifest)
    assert before['files_sha256'][key] != after['files_sha256'][key]
    assert before['content_sha256'] != after['content_sha256']


def test_qualification_policy_changes_source_fingerprint(tmp_path, monkeypatch):
    scripts=tmp_path/'dev'/'benchmarks';scripts.mkdir(parents=True)
    dispatcher=scripts/'run_coverage_benchmarks.py';dispatcher.write_text('# dispatcher\n')
    helper=scripts/'benchmark_qualification.py';helper.write_text('# policy1\n')
    manifest=scripts/'manifest.json';manifest.write_text('{}')
    monkeypatch.setattr(m,'ROOT',tmp_path);monkeypatch.setattr(m,'__file__',str(dispatcher))
    before=m.fingerprint(manifest)
    helper.write_text('# policy2\n')
    after=m.fingerprint(manifest)
    assert before['content_sha256'] != after['content_sha256']
    assert 'dev/benchmarks/benchmark_qualification.py' in after['files_sha256']
