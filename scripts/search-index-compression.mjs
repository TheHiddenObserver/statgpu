import { createHash } from 'node:crypto';
import { gzipSync, gunzipSync } from 'node:zlib';
import { gunzipSync as fallbackGunzip } from 'fflate';

const INDEX_ID = '/@localSearchIndex';
const DECODER_ID = '/@statgpuSearchIndexDecoder';
const LOCALES = new Set(['root', 'cn']);

// fflate does not validate the gzip checksum. Match native decompression's
// corruption rejection for the single gzip member emitted by this producer.
function gunzipFallback(compressed) {
  if (compressed.length < 18) throw new Error('Invalid gzip documentation search index');
  const output = fallbackGunzip(compressed);
  const table = new Uint32Array(256);
  for (let i = 0; i < 256; i += 1) {
    let value = i;
    for (let bit = 0; bit < 8; bit += 1) value = value & 1 ? 0xedb88320 ^ (value >>> 1) : value >>> 1;
    table[i] = value >>> 0;
  }
  let crc = 0xffffffff;
  for (const byte of output) crc = table[(crc ^ byte) & 0xff] ^ (crc >>> 8);
  const trailer = new DataView(compressed.buffer, compressed.byteOffset, compressed.byteLength);
  if (
    ((crc ^ 0xffffffff) >>> 0) !== trailer.getUint32(compressed.length - 8, true) ||
    (output.length >>> 0) !== trailer.getUint32(compressed.length - 4, true)
  ) {
    throw new Error('Corrupt gzip documentation search index');
  }
  return output;
}

// Emitted verbatim with the small fflate fallback above. Node-only imports
// must never enter the client bundle.
export async function decodeSearchIndex(payload) {
  if (
    payload?.encoding !== 'gzip-base64' ||
    typeof payload.data !== 'string' ||
    !Number.isSafeInteger(payload.uncompressedBytes) ||
    payload.uncompressedBytes < 0
  ) {
    throw new Error('Invalid compressed documentation search index');
  }
  const compressed = Uint8Array.from(atob(payload.data), character => character.charCodeAt(0));
  let bytes;
  if (typeof DecompressionStream === 'function') {
    const stream = new Blob([compressed]).stream().pipeThrough(new DecompressionStream('gzip'));
    bytes = await new Response(stream).arrayBuffer();
  } else {
    bytes = gunzipFallback(compressed);
  }
  if (bytes.byteLength !== payload.uncompressedBytes) {
    throw new Error('Documentation search index decompressed length mismatch');
  }
  return new TextDecoder('utf-8', { fatal: true }).decode(bytes);
}

export function compressSearchIndex(serialized) {
  const parsed = JSON.parse(serialized);
  if (!Number.isSafeInteger(parsed.documentCount) || !Array.isArray(parsed.index)) {
    throw new Error('Unexpected VitePress local-search JSON format');
  }
  const original = Buffer.from(serialized, 'utf8');
  const compressed = gzipSync(original, { level: 9 });
  if (!gunzipSync(compressed).equals(original)) {
    throw new Error('Documentation search compression did not preserve the original bytes');
  }
  return {
    encoding: 'gzip-base64',
    data: compressed.toString('base64'),
    uncompressedBytes: original.length,
    sha256: createHash('sha256').update(original).digest('hex'),
  };
}

function isFallbackModule(id) {
  return id.replace(/\\/g, '/').includes('/node_modules/fflate/');
}

export function compressedLocalSearchPlugin() {
  let ssr = false;
  return {
    name: 'statgpu-lossless-local-search',
    apply: 'build',
    enforce: 'post',
    configResolved(config) {
      ssr = Boolean(config.build.ssr);
    },
    resolveId(id) {
      if (id === DECODER_ID) return id;
    },
    load(id) {
      if (id === DECODER_ID) {
        return `import { gunzipSync as fallbackGunzip } from 'fflate';\n${gunzipFallback.toString()}\nexport ${decodeSearchIndex.toString()}`;
      }
    },
    outputOptions(options) {
      if (ssr) return;
      // Keep the decoder separately measurable instead of hiding its transfer
      // cost inside the main application chunk.
      const previous = options.manualChunks;
      if (previous && typeof previous !== 'function') {
        return {
          ...options,
          manualChunks: { ...previous, 'search-index-decoder': [DECODER_ID, 'fflate'] },
        };
      }
      return {
        ...options,
        manualChunks(id, context) {
          if (id === DECODER_ID || isFallbackModule(id)) return 'search-index-decoder';
          return previous?.(id, context);
        },
      };
    },
    generateBundle(_options, bundle) {
      if (ssr) return;
      const chunks = Object.values(bundle).filter(output => output.type === 'chunk');
      const decoder = chunks.filter(chunk => Object.hasOwn(chunk.modules, DECODER_ID));
      if (
        decoder.length !== 1 ||
        !Object.keys(decoder[0].modules).some(isFallbackModule) ||
        decoder[0].imports.length || decoder[0].dynamicImports.length ||
        chunks.some(chunk => chunk !== decoder[0] && Object.keys(chunk.modules).some(isFallbackModule))
      ) {
        throw new Error('Search decoder and every gzip fallback byte must share one self-contained budgeted chunk');
      }
    },
    transform(code, id) {
      if (ssr || !id.startsWith(INDEX_ID)) return;
      if (id === INDEX_ID) {
        const match = /^export default \{(.*)\}\s*;?$/s.exec(code);
        if (!match) throw new Error('Unexpected VitePress local-search loader format');
        const seen = new Set();
        for (const record of match[1].split(',')) {
          const item = /^"([^"]+)": \(\) => import\('@localSearchIndex([^']+)'\)$/.exec(record.trim());
          if (!item || item[1] !== item[2] || !LOCALES.has(item[1]) || seen.has(item[1])) {
            throw new Error('Unexpected VitePress local-search locale loader');
          }
          seen.add(item[1]);
        }
        if (seen.size !== LOCALES.size) {
          throw new Error('Expected English and Chinese documentation search indexes');
        }
        return {
          code: `import { decodeSearchIndex } from ${JSON.stringify(DECODER_ID)};\n` +
            `const originalLoaders = {${match[1]}};\n` +
            `export default Object.fromEntries(Object.entries(originalLoaders).map(([locale, load]) => {\n` +
            `  let pending;\n` +
            `  return [locale, () => pending ??= load().then(async module => ({ default: await decodeSearchIndex(module.default) })).catch(error => { pending = undefined; throw error; })];\n` +
            `}));\n`,
          map: null,
        };
      }
      if (!LOCALES.has(id.slice(INDEX_ID.length))) {
        throw new Error(`Unexpected VitePress local-search module: ${id}`);
      }
      const match = /^export default (".*")\s*;?$/s.exec(code);
      if (!match) throw new Error('Unexpected VitePress local-search index module format');
      const serialized = JSON.parse(match[1]);
      return { code: `export default ${JSON.stringify(compressSearchIndex(serialized))}`, map: null };
    },
  };
}
