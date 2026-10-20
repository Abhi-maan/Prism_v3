# Module 1 robustness benchmark (quick)

- Run: 2026-10-20 05:31:27 | git: 057c0e19+uncommitted-changes | path: app.run_duplicate_detection
- DB: 28409 entries | tracks: 25 ({'user_upload': 1, 'environment': 8, 'music': 8, 'speech': 8})
- Correct = matched AND matched_filename == true DB filename (top-1, reported as Duplicate/Edited Audio).

| Condition | Group | n | Top-1 acc | env | music | speech | user | wrong-match | mean ms | p95 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| duplicate | Duplicate | 25 | 100.0% | 8/8 | 8/8 | 8/8 | 1/1 | 0 | 26.8 | 33.1 |
| pitch_-4st | Pitch shift | 25 | 96.0% | 7/8 | 8/8 | 8/8 | 1/1 | 0 | 188.9 | 230.4 |
| pitch_-2st | Pitch shift | 25 | 96.0% | 7/8 | 8/8 | 8/8 | 1/1 | 0 | 117.7 | 198.5 |
| pitch_-1st | Pitch shift | 25 | 92.0% | 7/8 | 7/8 | 8/8 | 1/1 | 1 | 68.2 | 180.2 |
| pitch_+1st | Pitch shift | 25 | 96.0% | 7/8 | 8/8 | 8/8 | 1/1 | 0 | 62.9 | 197.2 |
| pitch_+2st | Pitch shift | 25 | 96.0% | 7/8 | 8/8 | 8/8 | 1/1 | 0 | 119.5 | 203.9 |
| pitch_+4st | Pitch shift | 25 | 92.0% | 6/8 | 8/8 | 8/8 | 1/1 | 0 | 173.8 | 202.3 |
| speed_-20pct | Speed change | 25 | 88.0% | 6/8 | 7/8 | 8/8 | 1/1 | 1 | 141.2 | 210.5 |
| speed_-10pct | Speed change | 25 | 92.0% | 7/8 | 7/8 | 8/8 | 1/1 | 1 | 79.6 | 199.4 |
| speed_-5pct | Speed change | 25 | 92.0% | 7/8 | 7/8 | 8/8 | 1/1 | 1 | 87.7 | 212.3 |
| speed_+5pct | Speed change | 25 | 96.0% | 7/8 | 8/8 | 8/8 | 1/1 | 0 | 69.1 | 191.4 |
| speed_+10pct | Speed change | 25 | 96.0% | 7/8 | 8/8 | 8/8 | 1/1 | 0 | 72.1 | 177.9 |
| speed_+20pct | Speed change | 25 | 92.0% | 6/8 | 8/8 | 8/8 | 1/1 | 0 | 113.0 | 184.9 |
| mp3_64kbps | Compression | 25 | 100.0% | 8/8 | 8/8 | 8/8 | 1/1 | 0 | 26.0 | 38.2 |
| mp3_128kbps | Compression | 25 | 100.0% | 8/8 | 8/8 | 8/8 | 1/1 | 0 | 24.5 | 33.5 |
| noise_snr10db | Noise | 25 | 52.0% | 2/8 | 6/8 | 5/8 | 0/1 | 0 | 111.4 | 235.9 |
| noise_snr20db | Noise | 25 | 72.0% | 4/8 | 7/8 | 7/8 | 0/1 | 0 | 79.9 | 222.8 |
| gain_-6db | Gain | 25 | 100.0% | 8/8 | 8/8 | 8/8 | 1/1 | 0 | 20.9 | 28.9 |
| gain_-3db | Gain | 25 | 100.0% | 8/8 | 8/8 | 8/8 | 1/1 | 0 | 20.8 | 28.7 |
| gain_+3db | Gain | 25 | 100.0% | 8/8 | 8/8 | 8/8 | 1/1 | 0 | 20.8 | 28.9 |
| gain_+6db | Gain | 25 | 100.0% | 8/8 | 8/8 | 8/8 | 1/1 | 0 | 20.9 | 29.2 |
| highpass_150hz | High-pass | 25 | 100.0% | 8/8 | 8/8 | 8/8 | 1/1 | 0 | 23.5 | 38.6 |
| highpass_200hz | High-pass | 25 | 96.0% | 7/8 | 8/8 | 8/8 | 1/1 | 0 | 23.2 | 31.4 |
| trim_start_3s | Trim | 23 | 91.3% | 7/8 | 7/8 | 6/6 | 1/1 | 1 | 21.1 | 31.1 |
| trim_start_10s | Trim | 9 | 88.9% | - | 7/8 | - | 1/1 | 1 | 29.9 | 38.3 |
| trim_end_3s | Trim | 23 | 100.0% | 8/8 | 8/8 | 6/6 | 1/1 | 0 | 17.7 | 29.5 |
| trim_end_10s | Trim | 9 | 100.0% | - | 8/8 | - | 1/1 | 0 | 24.5 | 35.0 |
| timeshift_cut_5s | Time shift | 15 | 100.0% | - | 8/8 | 6/6 | 1/1 | 0 | 23.3 | 29.7 |
| timeshift_cut_20s | Time shift | 9 | 100.0% | - | 8/8 | - | 1/1 | 0 | 27.7 | 29.9 |
| timeshift_pad_5s | Time shift | 25 | 100.0% | 8/8 | 8/8 | 8/8 | 1/1 | 0 | 23.4 | 36.6 |
| timeshift_pad_20s | Time shift | 25 | 100.0% | 8/8 | 8/8 | 8/8 | 1/1 | 0 | 24.3 | 35.6 |

Overall edited accuracy (all modified conditions pooled): 646/688 = 93.9%  | latency mean 68.6 ms, p95 207.1 ms

| Negatives (not in DB) | n | false positives | FPR |
|---|---:|---:|---:|
| librispeech_test_clean | 8 | 0 | 0.0% |
| local_music_mp3_segment | 8 | 0 | 0.0% |
| local_speech_mp3_segment | 4 | 0 | 0.0% |
| esc50_time_reversed | 8 | 1 | 12.5% |
| **all** | 28 | 1 | 3.6% |
