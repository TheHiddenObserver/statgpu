import { execFileSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

// These website outputs and dependencies are ignored, not versioned inputs.
// Keep workflow selectors and the independent policy cases in the tests aligned.
export const generatedArtifactRoots = Object.freeze([
  'node_modules',
  'frontend/node_modules',
  'frontend/test-results',
  'frontend/playwright-report',
  'frontend/dist',
  'frontend/public/data',
  'docs/assets/benchmarks',
  'docs/.vitepress/cache',
  'docs/.vitepress/.temp',
  '.site-dist',
]);

export function trackedGeneratedArtifacts(repositoryRoot) {
  // NUL separators preserve spaces, tabs, newlines, and Git-quoted filenames.
  // Inspect the index: a clean checkout can still contain committed artifacts.
  const tracked = execFileSync('git', ['ls-files', '--cached', '--full-name', '-z'], {
    cwd: repositoryRoot,
    encoding: 'utf8',
    maxBuffer: 32 * 1024 * 1024,
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  return tracked.split('\0').filter(filename => filename && generatedArtifactRoots.some(root =>
    filename === root || filename.startsWith(root + '/')));
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
  try {
    const artifacts = trackedGeneratedArtifacts(root);
    if (artifacts.length) {
      console.error('Generated website artifacts must not be tracked by Git:');
      for (const filename of artifacts) console.error(`  ${JSON.stringify(filename)}`);
      console.error('Remove these paths from the Git index; keep generated files and dependencies local.');
      process.exitCode = 1;
    } else {
      console.log('Verified no generated website artifacts are tracked by Git');
    }
  } catch (error) {
    console.error(`Unable to inspect tracked website artifacts: ${error.message}`);
    process.exitCode = 1;
  }
}
