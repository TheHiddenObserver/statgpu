import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('runtime_probe',Path(__file__).resolve().parents[1] / 'benchmarks/data_update/verify_runtime.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class RuntimePreflightTests(unittest.TestCase):
    def git(self,repo,*args):
        return {('rev-parse','HEAD'):m.PIN,('rev-parse','HEAD^{tree}'):m.TREE,
                ('status','--porcelain'):'',('rev-parse','--is-shallow-repository'):'false'}[args]
    def check(self,side_effect=None):
        with patch.object(m,'verify_tracked_source',return_value={}),patch.object(m,'git',side_effect=side_effect or self.git),patch.dict(os.environ,{k:'1' for k in m.THREAD_KEYS}):
            return m.verify(Path('/unused'),gpu=False)
    def test_source_only_never_claims_device_or_performance(self):
        r=self.check()
        self.assertFalse(r['performance_eligible']);self.assertFalse(r['gpu_probe_requested'])
        self.assertNotEqual(r['status'],'device_preflight_pass');self.assertNotIn('cupy',r)
    def test_wrong_head_rejected_before_gpu(self):
        def changed(repo,*args):return 'wrong' if args==('rev-parse','HEAD') else self.git(repo,*args)
        self.assertIn('Requires the exact clean non-shallow merged source pin',self.check(changed)['errors'])
    def test_dirty_rejected(self):
        def dirty(repo,*args):return ' M file' if args==('status','--porcelain') else self.git(repo,*args)
        self.assertEqual(self.check(dirty)['status'],'failed')
    def test_shallow_rejected(self):
        def shallow(repo,*args):return 'true' if args==('rev-parse','--is-shallow-repository') else self.git(repo,*args)
        self.assertEqual(self.check(shallow)['status'],'failed')
    def test_missing_thread_environment_rejected(self):
        with patch.object(m,'verify_tracked_source',return_value={}),patch.object(m,'git',side_effect=self.git),patch.dict(os.environ,{},clear=True):
            self.assertTrue(any('thread' in x for x in m.verify(Path('/unused'))['errors']))

class TrackedSourceTests(unittest.TestCase):
    def test_hidden_tracked_edits_do_not_pass_as_clean(self):
        for flag in ('--assume-unchanged', '--skip-worktree'):
            with self.subTest(flag=flag), tempfile.TemporaryDirectory() as temporary:
                repo = Path(temporary)
                def git(*args):
                    return subprocess.check_output(['git', '-C', str(repo), *args], stderr=subprocess.PIPE)
                git('init', '--quiet')
                source = repo / 'fixture.py'
                source.write_text('VALUE = 1\n', encoding='utf-8')
                git('add', 'fixture.py')
                git('-c', 'user.name=Synthetic Test', '-c', 'user.email=test@example.invalid',
                    'commit', '--quiet', '-m', 'synthetic fixture')
                self.assertEqual(set(m.verify_tracked_source(repo)), {'fixture.py'})
                git('update-index', flag, 'fixture.py')
                source.write_text('VALUE = 2\n', encoding='utf-8')
                self.assertEqual(git('status', '--porcelain').strip(), b'')
                with self.assertRaisesRegex(RuntimeError, 'Tracked bytes differ from HEAD'):
                    m.verify_tracked_source(repo)
if __name__=='__main__':unittest.main()
