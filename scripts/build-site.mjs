import { spawnSync } from 'node:child_process';

const npmCli = process.env.npm_execpath;
if (!npmCli) {
  throw new Error('site:build must be launched through npm so npm_execpath is set');
}

function run(command, args) {
  const result = spawnSync(command, args, {
    cwd: process.cwd(),
    env: process.env,
    stdio: 'inherit',
  });
  if (result.error) throw result.error;
  if (result.status !== 0) process.exit(result.status ?? 1);
}

run(process.execPath, [npmCli, 'run', 'docs:build']);
run(process.execPath, [npmCli, 'run', 'build', '--prefix', 'frontend']);
run(process.execPath, ['scripts/assemble-site.mjs']);
run(process.execPath, ['scripts/verify-site.mjs']);
