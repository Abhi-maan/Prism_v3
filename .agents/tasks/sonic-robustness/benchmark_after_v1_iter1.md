# Module 1 robustness benchmark (full)

- Run: 2026-10-20 06:32:28 | git: 057c0e19+uncommitted-changes | path: app.run_duplicate_detection
- DB: 28409 entries | tracks: 103 ({'user_upload': 1, 'environment': 34, 'music': 34, 'speech': 34})
- Correct = matched AND matched_filename == true DB filename (top-1, reported as Duplicate/Edited Audio).

| Condition | Group | n | Top-1 acc | env | music | speech | user | wrong-match | mean ms | p95 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| duplicate | Duplicate | 103 | 99.0% | 33/34 | 34/34 | 34/34 | 1/1 | 0 | 20.6 | 31.5 |
| pitch_-4st | Pitch shift | 103 | 84.5% | 23/34 | 33/34 | 30/34 | 1/1 | 1 | 193.0 | 247.9 |
| pitch_-2st | Pitch shift | 103 | 87.4% | 25/34 | 33/34 | 31/34 | 1/1 | 0 | 137.0 | 227.8 |
| pitch_-1st | Pitch shift | 103 | 85.4% | 23/34 | 33/34 | 31/34 | 1/1 | 1 | 104.1 | 230.2 |
| pitch_+1st | Pitch shift | 103 | 85.4% | 23/34 | 33/34 | 31/34 | 1/1 | 0 | 83.3 | 229.6 |
| pitch_+2st | Pitch shift | 103 | 88.3% | 25/34 | 34/34 | 31/34 | 1/1 | 0 | 123.0 | 217.8 |
| pitch_+4st | Pitch shift | 103 | 83.5% | 23/34 | 31/34 | 31/34 | 1/1 | 1 | 177.1 | 229.7 |
| speed_-20pct | Speed change | 103 | 85.4% | 25/34 | 31/34 | 31/34 | 1/1 | 1 | 126.1 | 227.1 |
| speed_-10pct | Speed change | 103 | 86.4% | 24/34 | 33/34 | 31/34 | 1/1 | 1 | 73.0 | 225.2 |
| speed_-5pct | Speed change | 103 | 86.4% | 23/34 | 34/34 | 31/34 | 1/1 | 2 | 68.9 | 224.1 |
| speed_+5pct | Speed change | 103 | 90.3% | 28/34 | 33/34 | 31/34 | 1/1 | 0 | 63.7 | 220.5 |
| speed_+10pct | Speed change | 103 | 87.4% | 25/34 | 34/34 | 30/34 | 1/1 | 1 | 76.3 | 207.2 |
| speed_+20pct | Speed change | 103 | 81.5% | 22/34 | 33/34 | 28/34 | 1/1 | 0 | 116.8 | 206.0 |
| mp3_64kbps | Compression | 103 | 99.0% | 33/34 | 34/34 | 34/34 | 1/1 | 0 | 22.2 | 33.0 |
| mp3_128kbps | Compression | 103 | 98.1% | 33/34 | 33/34 | 34/34 | 1/1 | 1 | 20.5 | 29.4 |
| noise_snr10db | Noise | 103 | 42.7% | 13/34 | 16/34 | 15/34 | 0/1 | 0 | 124.9 | 233.5 |
| noise_snr20db | Noise | 103 | 78.6% | 24/34 | 29/34 | 28/34 | 0/1 | 1 | 53.7 | 212.4 |
| gain_-6db | Gain | 103 | 99.0% | 33/34 | 34/34 | 34/34 | 1/1 | 0 | 18.0 | 29.2 |
| gain_-3db | Gain | 103 | 99.0% | 33/34 | 34/34 | 34/34 | 1/1 | 0 | 17.5 | 28.5 |
| gain_+3db | Gain | 103 | 99.0% | 33/34 | 34/34 | 34/34 | 1/1 | 0 | 17.5 | 28.7 |
| gain_+6db | Gain | 103 | 99.0% | 33/34 | 34/34 | 34/34 | 1/1 | 0 | 17.5 | 29.0 |
| highpass_150hz | High-pass | 103 | 98.1% | 33/34 | 33/34 | 34/34 | 1/1 | 1 | 20.2 | 28.6 |
| highpass_200hz | High-pass | 103 | 99.0% | 33/34 | 34/34 | 34/34 | 1/1 | 0 | 20.1 | 26.5 |
| trim_start_3s | Trim | 92 | 85.9% | 22/34 | 33/34 | 23/23 | 1/1 | 1 | 22.5 | 55.5 |
| trim_start_10s | Trim | 35 | 97.1% | - | 33/34 | - | 1/1 | 1 | 23.8 | 29.3 |
| trim_end_3s | Trim | 92 | 95.7% | 30/34 | 34/34 | 23/23 | 1/1 | 0 | 16.7 | 34.8 |
| trim_end_10s | Trim | 35 | 100.0% | - | 34/34 | - | 1/1 | 0 | 21.4 | 33.9 |
| timeshift_cut_5s | Time shift | 48 | 100.0% | - | 34/34 | 13/13 | 1/1 | 0 | 22.1 | 28.6 |
| timeshift_cut_20s | Time shift | 35 | 100.0% | - | 34/34 | - | 1/1 | 0 | 24.0 | 28.0 |
| timeshift_pad_5s | Time shift | 103 | 99.0% | 33/34 | 34/34 | 34/34 | 1/1 | 0 | 20.0 | 31.6 |
| timeshift_pad_20s | Time shift | 103 | 99.0% | 33/34 | 34/34 | 34/34 | 1/1 | 0 | 20.5 | 31.5 |

Overall edited accuracy (all modified conditions pooled): 2525/2809 = 89.9%  | latency mean 68.7 ms, p95 220.1 ms

| Negatives (not in DB) | n | false positives | FPR |
|---|---:|---:|---:|
| librispeech_test_clean | 40 | 0 | 0.0% |
| local_music_mp3_segment | 18 | 1 | 5.6% |
| local_speech_mp3_segment | 20 | 0 | 0.0% |
| esc50_time_reversed | 40 | 0 | 0.0% |
| **all** | 118 | 1 | 0.9% |
