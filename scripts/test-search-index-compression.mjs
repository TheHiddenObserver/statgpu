import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { test } from 'node:test';
import MiniSearch from 'minisearch';
import { compressSearchIndex, compressedLocalSearchPlugin, decodeSearchIndex } from './search-index-compression.mjs';

const options = { fields: ['title', 'titles', 'text'], storeFields: ['title', 'titles'] };
function fixture() {
  const index = new MiniSearch(options);
  index.addAll([
    { id: '/statgpu/en/reference/linear-model-api#ridge', title: 'Ridge', titles: ['Linear-model API'], text: 'sample_weight alpha fit_intercept covariance naïve λ' },
    { id: '/statgpu/cn/reference/linear-model-api#ridge', title: 'Ridge', titles: ['线性模型 API'], text: '样本权重 协方差 参数选择 数学 λ 😀' },
  ]);
  return JSON.stringify(index);
}

async function emittedModule(code) {
  return import(`data:text/javascript;base64,${Buffer.from(code.replaceAll("from 'fflate'", `from ${JSON.stringify(import.meta.resolve('fflate'))}`)).toString('base64')}`);
}

test('lossless deterministic compression preserves every byte and search result', async () => {
  const original = fixture();
  const payload = compressSearchIndex(original);
  assert.deepEqual(payload, compressSearchIndex(original));
  const decoded = await decodeSearchIndex(payload);
  assert.equal(decoded, original);
  assert.equal(createHash('sha256').update(decoded).digest('hex'), payload.sha256);
  for (const query of ['sample_weight', 'Ridge', '样本权重', '协方差']) {
    assert.deepEqual(MiniSearch.loadJSON(decoded, options).search(query), MiniSearch.loadJSON(original, options).search(query));
    assert.ok(MiniSearch.loadJSON(decoded, options).search(query).length);
  }
});

test('the exact emitted browser decoder preserves Unicode and reports invalid input', async () => {
  const plugin = compressedLocalSearchPlugin();
  const decoder = await emittedModule(plugin.load('/@statgpuSearchIndexDecoder'));
  assert.equal(await decoder.decodeSearchIndex(compressSearchIndex(fixture())), fixture());
  await assert.rejects(decoder.decodeSearchIndex({}), /Invalid compressed/);
  const wrongLength = { ...compressSearchIndex(fixture()), uncompressedBytes: 1 };
  await assert.rejects(decoder.decodeSearchIndex(wrongLength), /length mismatch/);
});

test('browsers without native decompression retain complete Unicode search', async () => {
  const original = globalThis.DecompressionStream;
  try {
    globalThis.DecompressionStream = undefined;
    const plugin = compressedLocalSearchPlugin();
    const decoder = await emittedModule(plugin.load('/@statgpuSearchIndexDecoder'));
    const serialized = fixture();
    const decoded = await decoder.decodeSearchIndex(compressSearchIndex(serialized));
    assert.equal(decoded, serialized);
    for (const query of ['Ridge', 'sample_weight', '样本权重', '协方差']) {
      const before = MiniSearch.loadJSON(serialized, options).search(query);
      assert.ok(before.length);
      assert.deepEqual(MiniSearch.loadJSON(decoded, options).search(query), before);
    }
  } finally {
    globalThis.DecompressionStream = original;
  }
});

test('virtual locale modules retain original data and the loader interface', async () => {
  const plugin = compressedLocalSearchPlugin();
  const original = fixture();
  for (const locale of ['root', 'cn']) {
    const transformed = plugin.transform(`export default ${JSON.stringify(original)}`, `/@localSearchIndex${locale}`);
    const module = await emittedModule(transformed.code);
    assert.equal(await decodeSearchIndex(module.default), original);
  }
  const loader = plugin.transform(`export default {"root": () => import('@localSearchIndexroot'),"cn": () => import('@localSearchIndexcn')}`, '/@localSearchIndex');
  const payload = JSON.stringify(compressSearchIndex(original));
  const runnable = loader.code
    .replace('import { decodeSearchIndex } from "/@statgpuSearchIndexDecoder";', plugin.load('/@statgpuSearchIndexDecoder'))
    .replaceAll(/import\('@localSearchIndex(?:root|cn)'\)/g, `Promise.resolve({ default: ${payload} })`);
  const { default: loaders } = await emittedModule(runnable);
  assert.deepEqual(Object.keys(loaders), ['root', 'cn']);
  for (const load of Object.values(loaders)) {
    assert.equal(load(), load(), 'prefetch and search share one cached decoding promise');
    assert.deepEqual(await load(), { default: original });
  }
});

test('unexpected upstream formats fail closed and SSR stays unchanged', () => {
  const plugin = compressedLocalSearchPlugin();
  assert.throws(() => plugin.transform('export default {}', '/@localSearchIndex'), /locale loader/);
  assert.throws(() => plugin.transform('export default "{}"', '/@localSearchIndexroot'), /JSON format/);
  assert.throws(() => plugin.transform('export default "{}"', '/@localSearchIndexfr'), /Unexpected/);
  plugin.configResolved({ build: { ssr: true } });
  assert.equal(plugin.transform('export default {}', '/@localSearchIndex'), undefined);
  assert.equal(plugin.outputOptions({}), undefined);
});


test('native and fallback decoders reject malformed streams and checksum corruption', async () => {
  const native = globalThis.DecompressionStream;
  try {
    for (const capability of [native, undefined]) {
      globalThis.DecompressionStream = capability;
      const payload = compressSearchIndex(fixture());
      await assert.rejects(decodeSearchIndex({ ...payload, data: Buffer.from('invalid gzip data').toString('base64') }));
      const corrupted = Buffer.from(payload.data, 'base64');
      corrupted[corrupted.length - 8] ^= 0xff;
      await assert.rejects(decodeSearchIndex({ ...payload, data: corrupted.toString('base64') }));
    }
  } finally {
    globalThis.DecompressionStream = native;
  }
});

test('a failed decoding promise is cleared so the next search can retry', async () => {
  const plugin = compressedLocalSearchPlugin();
  const original = fixture();
  const transformed = plugin.transform(`export default {"root": () => import('@localSearchIndexroot'),"cn": () => import('@localSearchIndexcn')}`, '/@localSearchIndex');
  const payload = JSON.stringify(compressSearchIndex(original));
  const runnable = `let attempts = 0;\n` + transformed.code
    .replace('import { decodeSearchIndex } from "/@statgpuSearchIndexDecoder";', plugin.load('/@statgpuSearchIndexDecoder'))
    .replaceAll(/import\('@localSearchIndex(?:root|cn)'\)/g, `Promise.resolve({ default: ++attempts === 1 ? {} : ${payload} })`);
  const { default: loaders } = await emittedModule(runnable);
  const first = loaders.root();
  await assert.rejects(first, /Invalid compressed/);
  const second = loaders.root();
  assert.notEqual(second, first);
  assert.deepEqual(await second, { default: original });
  assert.equal(loaders.root(), second);
});

test('every fallback module is assigned to the measured decoder chunk', () => {
  const plugin = compressedLocalSearchPlugin();
  const output = plugin.outputOptions({});
  assert.equal(output.manualChunks('/@statgpuSearchIndexDecoder'), 'search-index-decoder');
  assert.equal(output.manualChunks('/project/node_modules/fflate/esm/browser.js'), 'search-index-decoder');
  const decoder = {
    type: 'chunk', modules: { '/@statgpuSearchIndexDecoder': {}, '/project/node_modules/fflate/esm/browser.js': {} },
    imports: [], dynamicImports: [],
  };
  plugin.generateBundle({}, { decoder });
  assert.throws(() => plugin.generateBundle({}, { decoder: { ...decoder, imports: ['hidden-fallback.js'] } }), /every gzip fallback byte/);
  assert.throws(() => plugin.generateBundle({}, { decoder, hidden: { type: 'chunk', modules: { '/project/node_modules/fflate/esm/browser.js': {} } } }), /every gzip fallback byte/);
});
