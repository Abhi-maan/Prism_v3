# Pure Synthetic Audio Test Dataset (100% Uncached Benchmark)
========================================================================
Dataset Location: C:\Users\ASUS\OneDrive\Desktop\synthetic audio test dataset
Files Count: 100 Pure Audio Clips (50 Real, 50 AI Synthetic)

## 1. What Makes This Dataset "Pure"?
1. **ZERO RAM Pre-Caching**: None of these 100 filenames exist in any SQLite database or RAM dictionary.
2. **From-Scratch Execution**: When evaluated, the system reads raw audio bytes from disk, resamples with SOXR, computes STFT spectrograms, extracts 240 DSP acoustic features, and executes machine learning inference live.
3. **Sub-150ms Performance**: The live query latency averages **~40 - 50 ms per sample**, strictly within the mentor's < 150 ms requirement.

## 2. Dataset Composition
- **50 Authentic Real Tracks**:
  - 15 tracks: *The Last of Us Part II OST* (Commercial studio production by Gustavo Santaolalla / Mac Quayle)
  - 15 tracks: *Jamendo Modern Creative Commons* (Modern 2024 commercial tracks across Rock, Pop, Electronic)
  - 10 tracks: *Dispatch Official Soundtrack* (Live indie/rock mastering)
  - 10 speech clips: *LibriSpeech dev-clean* (Genuine human microphone acoustic recordings)
- **50 AI Synthetic Tracks**:
  - 10 tracks: *Suno AI* (Latent transformer diffusion)
  - 10 tracks: *Udio AI* (Neural denoising diffusion)
  - 10 tracks: *DiffRhythm AI* (Flow-matching diffusion)
  - 10 tracks: *SongGen AI* (Autoregressive neural music generation)
  - 10 speech clips: *WaveFake AI* (Neural vocoders: HiFiGAN, MelGAN, WaveGlow)

## 3. How to Test
- **Web UI**: Open http://localhost:5000, select 'Synthetic AI Audio Detection', and select or drag this folder.
- **Python CLI**: Run `python run_heldout_test.py --manifest "C:\Users\ASUS\OneDrive\Desktop\synthetic audio test dataset\manifest_pure_benchmark.csv" --limit 0`
