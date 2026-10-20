# Module 1 robustness benchmark (full)

- Run: 2026-10-20 07:09:48 | git: 057c0e19+uncommitted-changes | path: app.run_duplicate_detection
- DB: 28409 entries | tracks: 103 ({'user_upload': 1, 'environment': 34, 'music': 34, 'speech': 34})
- Correct = matched AND matched_filename == true DB filename (top-1, reported as Duplicate/Edited Audio).

| Condition | Group | n | Top-1 acc | env | music | speech | user | wrong-match | mean ms | p95 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| duplicate | Duplicate | 103 | 97.1% | 31/34 | 34/34 | 34/34 | 1/1 | 0 | 24.8 | 32.4 |
| pitch_-4st | Pitch shift | 103 | 82.5% | 21/34 | 33/34 | 30/34 | 1/1 | 0 | 196.2 | 250.2 |
| pitch_-2st | Pitch shift | 103 | 84.5% | 22/34 | 33/34 | 31/34 | 1/1 | 0 | 140.7 | 232.7 |
| pitch_-1st | Pitch shift | 103 | 83.5% | 21/34 | 33/34 | 31/34 | 1/1 | 0 | 110.5 | 231.7 |
| pitch_+1st | Pitch shift | 103 | 83.5% | 21/34 | 33/34 | 31/34 | 1/1 | 0 | 88.4 | 223.5 |
| pitch_+2st | Pitch shift | 103 | 85.4% | 22/34 | 34/34 | 31/34 | 1/1 | 0 | 126.8 | 221.0 |
| pitch_+4st | Pitch shift | 103 | 81.5% | 21/34 | 31/34 | 31/34 | 1/1 | 0 | 180.3 | 231.1 |
| speed_-20pct | Speed change | 103 | 83.5% | 23/34 | 31/34 | 31/34 | 1/1 | 0 | 134.5 | 230.5 |
| speed_-10pct | Speed change | 103 | 85.4% | 23/34 | 33/34 | 31/34 | 1/1 | 0 | 81.2 | 225.8 |
| speed_-5pct | Speed change | 103 | 85.4% | 22/34 | 34/34 | 31/34 | 1/1 | 0 | 76.6 | 231.5 |
| speed_+5pct | Speed change | 103 | 86.4% | 24/34 | 33/34 | 31/34 | 1/1 | 0 | 71.8 | 222.4 |
| speed_+10pct | Speed change | 103 | 85.4% | 23/34 | 34/34 | 30/34 | 1/1 | 0 | 82.1 | 209.8 |
| speed_+20pct | Speed change | 103 | 80.6% | 21/34 | 33/34 | 28/34 | 1/1 | 0 | 121.5 | 209.0 |
| mp3_64kbps | Compression | 103 | 97.1% | 31/34 | 34/34 | 34/34 | 1/1 | 0 | 26.5 | 35.3 |
| mp3_128kbps | Compression | 103 | 96.1% | 31/34 | 33/34 | 34/34 | 1/1 | 1 | 24.7 | 32.1 |
| noise_snr10db | Noise | 103 | 41.8% | 12/34 | 16/34 | 15/34 | 0/1 | 0 | 128.4 | 239.9 |
| noise_snr20db | Noise | 103 | 77.7% | 23/34 | 29/34 | 28/34 | 0/1 | 1 | 57.1 | 224.0 |
| gain_-6db | Gain | 103 | 97.1% | 31/34 | 34/34 | 34/34 | 1/1 | 0 | 22.2 | 31.1 |
| gain_-3db | Gain | 103 | 97.1% | 31/34 | 34/34 | 34/34 | 1/1 | 0 | 21.8 | 30.4 |
| gain_+3db | Gain | 103 | 97.1% | 31/34 | 34/34 | 34/34 | 1/1 | 0 | 21.8 | 30.9 |
| gain_+6db | Gain | 103 | 97.1% | 31/34 | 34/34 | 34/34 | 1/1 | 0 | 21.7 | 30.7 |
| highpass_150hz | High-pass | 103 | 95.2% | 30/34 | 33/34 | 34/34 | 1/1 | 1 | 26.1 | 36.6 |
| highpass_200hz | High-pass | 103 | 96.1% | 30/34 | 34/34 | 34/34 | 1/1 | 0 | 25.9 | 29.8 |
| trim_start_3s | Trim | 92 | 85.9% | 22/34 | 33/34 | 23/23 | 1/1 | 1 | 22.9 | 55.6 |
| trim_start_10s | Trim | 35 | 97.1% | - | 33/34 | - | 1/1 | 1 | 24.7 | 29.7 |
| trim_end_3s | Trim | 92 | 95.7% | 30/34 | 34/34 | 23/23 | 1/1 | 0 | 17.2 | 35.2 |
| trim_end_10s | Trim | 35 | 100.0% | - | 34/34 | - | 1/1 | 0 | 22.2 | 34.3 |
| timeshift_cut_5s | Time shift | 48 | 100.0% | - | 34/34 | 13/13 | 1/1 | 0 | 23.0 | 28.4 |
| timeshift_cut_20s | Time shift | 35 | 97.1% | - | 33/34 | - | 1/1 | 0 | 30.2 | 32.7 |
| timeshift_pad_5s | Time shift | 103 | 97.1% | 31/34 | 34/34 | 34/34 | 1/1 | 0 | 24.3 | 34.7 |
| timeshift_pad_20s | Time shift | 103 | 97.1% | 31/34 | 34/34 | 34/34 | 1/1 | 0 | 24.9 | 34.4 |

Overall edited accuracy (all modified conditions pooled): 2475/2809 = 88.1%  | latency mean 73.2 ms, p95 223.6 ms

| Negatives (not in DB) | n | false positives | FPR |
|---|---:|---:|---:|
| librispeech_test_clean | 40 | 0 | 0.0% |
| local_music_mp3_segment | 18 | 0 | 0.0% |
| local_speech_mp3_segment | 20 | 0 | 0.0% |
| esc50_time_reversed | 40 | 0 | 0.0% |
| **all** | 118 | 0 | 0.0% |
