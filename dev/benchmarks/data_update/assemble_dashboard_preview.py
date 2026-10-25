#!/usr/bin/env python3
"""Make a fresh offline dashboard copy with a validated diagnostic bundle. Never deploy."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

FILES = ('benchmark_data.json', 'parse_report.json', 'source_inventory.json')

def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result: raise ValueError('Duplicate JSON key: ' + key)
            result[key] = value
        return result
    def number(value):
        parsed = float(value)
        if not math.isfinite(parsed): raise ValueError('Nonfinite JSON number')
        return parsed
    def invalid(value): raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=pairs,
                      parse_float=number, parse_constant=invalid)

def assemble(repo, bundle, assets, destination):
    if any(p.is_symlink() for p in (repo,bundle,assets,destination)):
        raise ValueError('Symlink root paths are not supported')
    repo, bundle, assets, destination = [p.resolve() for p in (repo,bundle,assets,destination)]
    if destination.exists() or destination.is_relative_to(repo) or not destination.parent.is_dir():
        raise ValueError('Output must be fresh, outside checkout, with an existing parent')
    if destination.is_relative_to(assets) or destination.is_relative_to(bundle):
        raise ValueError('Output must not nest inside inputs')
    if not (assets/'index.html').is_file(): raise ValueError('Built assets must contain index.html')
    paths=list(assets.rglob('*'))
    if any(p.is_symlink() for p in paths): raise ValueError('Symlink assets are not supported')
    if sum(p.stat().st_size for p in paths if p.is_file()) > 128*1024*1024:
        raise ValueError('Built assets exceed the 128 MiB local preview limit')
    documents={}
    for name in FILES:
        p=bundle/name
        if p.is_symlink() or not p.is_file() or p.stat().st_size>64*1024*1024:
            raise ValueError('Missing/oversized/nonregular bundle file: '+name)
        documents[name]=read_json(p)
    data,report,inventory=[documents[n] for n in FILES]
    if not all(isinstance(x,dict) for x in (data,report,inventory)):
        raise ValueError('Bundle JSON roots must be objects')
    if report.get('report_version') != '2.0' or inventory.get('inventory_version') != '2.0':
        raise ValueError('Unsupported report/inventory contract')
    for key in ('files_seen','files_parsed','files_skipped','runs_generated'):
        if type(report.get(key)) is not int or report[key] < 0: raise ValueError('Invalid report count: '+key)
    if not isinstance(report.get('issues'),list) or not all(isinstance(x,dict) for x in report['issues']):
        raise ValueError('Invalid report issues')
    for key in ('discovered_json_artifacts','classified_candidate_sources','eligible_sources','registered_sources','available_registered_sources','parsed_registered_sources','eligible_unregistered_sources','not_canonical_ready_sources','historical_or_excluded_sources','unclassified_artifacts'):
        if type(inventory.get(key)) is not int or inventory[key] < 0: raise ValueError('Invalid inventory count: '+key)
    gid=data.get('meta',{}).get('generation_id')
    if not gid or report.get('generation_id')!=gid or inventory.get('generation_id')!=gid:
        raise ValueError('Mixed generation IDs')
    if report.get('runs_generated')!=len(data.get('runs',[])):
        raise ValueError('Run count mismatch')
    sys.path.insert(0,str(repo))
    from dev.benchmarks.frontend_data.cli import validate_output,validate_against_schema,validate_semantic
    from dev.benchmarks.generate_benchmark_data import _assign_bundle_generation_id
    rebuilt = copy.deepcopy([data, report, inventory])
    _assign_bundle_generation_id(*rebuilt)
    if rebuilt[0]['meta']['generation_id'] != gid: raise ValueError('Bundle content digest mismatch')
    errors=validate_output(data)+validate_against_schema(data)+validate_semantic(data)
    if errors:raise ValueError('Invalid dashboard bundle: '+'; '.join(errors[:10]))
    for row in data.get('runs',[]):
        evidence=row.get('evidence',{})
        if ('speedup' in row.get('metrics',{}) or evidence.get('qualification') != 'diagnostic_only'
                or evidence.get('performance_eligible') is not False
                or evidence.get('canonical_dashboard_ready') is not False
                or evidence.get('machine_verified_exclusivity') is not False):
            raise ValueError('This assembler accepts diagnostic bundles only, not performance publication')
    shutil.copytree(assets,destination)
    try:
        target=destination/'data';target.mkdir(exist_ok=True)
        for name in FILES:shutil.copyfile(bundle/name,target/name)
        note={'generation_id':gid,'local_preview_only':True,'published':False,
              'input_sha256':{n:hashlib.sha256((bundle/n).read_bytes()).hexdigest() for n in FILES}}
        (destination/'PREVIEW_ONLY.json').write_text(json.dumps(note,indent=2)+'\n')
    except BaseException:
        # Preserve the new partial directory for diagnosis; do not delete user data.
        raise
    return destination

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('repo','bundle','built-assets','output-dir'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    try: print(assemble(a.repo,a.bundle,a.built_assets,a.output_dir))
    except (ValueError,OSError,TypeError,KeyError) as e:p.error(str(e))
if __name__=='__main__':main()
