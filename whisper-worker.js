// whisper-worker.js
// Chạy Whisper AI trong Web Worker để không block UI

import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2';

env.allowLocalModels = false;
env.useBrowserCache = true;

let transcriberCache = {};
let cancelled = false;

self.onmessage = async (e) => {
  const { type, payload } = e.data;

  if (type === 'cancel') {
    cancelled = true;
    return;
  }

  if (type === 'transcribe') {
    cancelled = false;
    const { audio, modelId, lang } = payload;

    try {
      // ═══════════════════════════════════════════════
      // 1. Load model (cache lại cho lần sau)
      // ═══════════════════════════════════════════════
      if (!transcriberCache[modelId]) {
        self.postMessage({ type: 'status', payload: { phase: 'load', percent: 15, label: 'Đang kết nối tới HuggingFace...' } });

        transcriberCache[modelId] = await pipeline('automatic-speech-recognition', modelId, {
          quantized: true,
          progress_callback: (p) => {
            if (cancelled) throw new Error('CANCELLED');
            self.postMessage({ type: 'model_progress', payload: p });
          },
        });
      }
      if (cancelled) throw new Error('CANCELLED');

      // ═══════════════════════════════════════════════
      // 2. Chạy inference
      // ═══════════════════════════════════════════════
      const transcriber = transcriberCache[modelId];
      const totalSec = audio.length / 16000;
      let processedSec = 0;

      self.postMessage({ type: 'status', payload: { phase: 'infer', percent: 40, label: 'Bắt đầu nhận dạng giọng nói...' } });

      const opts = {
        chunk_length_s: 30,
        stride_length_s: 5,
        return_timestamps: true,
        task: 'transcribe',
        chunk_callback: (chunk) => {
          if (cancelled) throw new Error('CANCELLED');
          // chunk là Float32Array của đoạn vừa xử lý
          const chunkSec = Math.min(30, chunk.length / 16000);
          processedSec += chunkSec;
          self.postMessage({
            type: 'infer_progress',
            payload: { processedSec, totalSec },
          });
        },
      };

      if (lang && lang !== 'auto') opts.language = lang;

      const result = await transcriber(audio, opts);
      if (cancelled) throw new Error('CANCELLED');

      self.postMessage({
        type: 'done',
        payload: { result, totalSec },
      });
    } catch (err) {
      if (err.message === 'CANCELLED' || cancelled) {
        self.postMessage({ type: 'cancelled' });
      } else {
        self.postMessage({ type: 'error', payload: { message: err.message || String(err) } });
      }
    }
  }
};
