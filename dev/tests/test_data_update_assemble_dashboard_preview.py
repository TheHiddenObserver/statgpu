import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
spec=importlib.util.spec_from_file_location('assemble_preview',Path(__file__).resolve().parents[1] / 'benchmarks/data_update/assemble_dashboard_preview.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class AssembleTests(unittest.TestCase):
    def test_json_rejects_duplicate_and_nonfinite(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.json'
            for text in ('{"x":1,"x":2}','{"x":1e999}','{"x":NaN}'):
                p.write_text(text)
                with self.assertRaises(ValueError):m.read_json(p)
    def test_existing_destination_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with self.assertRaisesRegex(ValueError,'fresh'):m.assemble(root/'repo',root/'bundle',root/'assets',root)
    def test_symlink_input_root_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/'assets').mkdir();(root/'alias').symlink_to(root/'assets',target_is_directory=True)
            with self.assertRaisesRegex(ValueError,'Symlink'):m.assemble(root/'repo',root/'bundle',root/'alias',root/'out')
    def test_malformed_bundle_roots_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'assets').mkdir();(root/'assets'/'index.html').write_text('fixture')
            (root/'bundle').mkdir()
            for name in m.FILES:(root/'bundle'/name).write_text('[]')
            with self.assertRaisesRegex(ValueError,'objects'):m.assemble(root/'repo',root/'bundle',root/'assets',root/'out')
    def test_truncated_report_contract_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'assets').mkdir();(root/'assets'/'index.html').write_text('fixture');(root/'bundle').mkdir()
            for name in m.FILES:(root/'bundle'/name).write_text(json.dumps({'generation_id':'x'}))
            with self.assertRaisesRegex(ValueError,'contract'):m.assemble(root/'repo',root/'bundle',root/'assets',root/'out')
if __name__=='__main__':unittest.main()
