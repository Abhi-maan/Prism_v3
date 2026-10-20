# Module 1 robustness benchmark (quick)

- Run: 2026-10-20 07:19:18 | git: 057c0e19+uncommitted-changes | path: POST /api/check (Flask test client)
- DB: 28409 entries | tracks: 25 ({'user_upload': 1, 'environment': 8, 'music': 8, 'speech': 8})
- Correct = matched AND matched_filename == true DB filename (top-1, reported as Duplicate/Edited Audio).

| Condition | Group | n | Top-1 acc | env | music | speech | user | wrong-match | mean ms | p95 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| duplicate | Duplicate | 25 | 100.0% | 8/8 | 8/8 | 8/8 | 1/1 | 0 | 28.1 | 37.5 |
| pitch_-4st | Pitch shift | 25 | 88.0% | 5/8 | 8/8 | 8/8 | 1/1 | 0 | 207.3 | 265.7 |
| pitch_-2st | Pitch shift | 25 | 92.0% | 6/8 | 8/8 | 8/8 | 1/1 | 0 | 148.6 | 236.7 |
| pitch_-1st | Pitch shift | 25 | 88.0% | 5/8 | 8/8 | 8/8 | 1/1 | 0 | 97.6 | 228.8 |
| pitch_+1st | Pitch shift | 25 | 88.0% | 5/8 | 8/8 | 8/8 | 1/1 | 0 | 88.8 | 232.1 |
| pitch_+2st | Pitch shift | 25 | 92.0% | 6/8 | 8/8 | 8/8 | 1/1 | 0 | 134.2 | 221.1 |
| pitch_+4st | Pitch shift | 25 | 80.0% | 5/8 | 6/8 | 8/8 | 1/1 | 0 | 189.2 | 250.6 |
| speed_-20pct | Speed change | 25 | 88.0% | 5/8 | 8/8 | 8/8 | 1/1 | 0 | 141.0 | 226.9 |
| speed_-10pct | Speed change | 25 | 88.0% | 5/8 | 8/8 | 8/8 | 1/1 | 0 | 87.4 | 226.5 |
| speed_-5pct | Speed change | 25 | 88.0% | 5/8 | 8/8 | 8/8 | 1/1 | 0 | 77.8 | 232.9 |
| speed_+5pct | Speed change | 25 | 88.0% | 5/8 | 8/8 | 8/8 | 1/1 | 0 | 59.7 | 228.4 |
| speed_+10pct | Speed change | 25 | 88.0% | 5/8 | 8/8 | 8/8 | 1/1 | 0 | 81.4 | 225.2 |
| speed_+20pct | Speed change | 25 | 88.0% | 5/8 | 8/8 | 8/8 | 1/1 | 0 | 128.2 | 221.5 |
| mp3_64kbps | Compression | 25 | 100.0% | 8/8 | 8/8 | 8/8 | 1/1 | 0 | 28.4 | 52.2 |
| mp3_128kbps | Compression | 25 | 96.0% | 8/8 | 7/8 | 8/8 | 1/1 | 1 | 23.9 | 31.2 |
| noise_snr10db | Noise | 25 | 40.0% | 2/8 | 4/8 | 4/8 | 0/1 | 0 | 142.8 | 247.8 |
| noise_snr20db | Noise | 25 | 80.0% | 5/8 | 7/8 | 8/8 | 0/1 | 0 | 73.8 | 234.1 |
| gain_-6db | Gain | 25 | 100.0% | 8/8 | 8/8 | 8/8 | 1/1 | 0 | 24.5 | 37.9 |
| gain_-3db | Gain | 25 | 100.0% | 8/8 | 8/8 | 8/8 | 1/1 | 0 | 24.5 | 36.9 |
| gain_+3db | Gain | 25 | 100.0% | 8/8 | 8/8 | 8/8 | 1/1 | 0 | 24.7 | 37.2 |
| gain_+6db | Gain | 25 | 100.0% | 8/8 | 8/8 | 8/8 | 1/1 | 0 | 24.1 | 36.9 |
| highpass_150hz | High-pass | 25 | 96.0% | 8/8 | 7/8 | 8/8 | 1/1 | 1 | 29.8 | 67.0 |
| highpass_200hz | High-pass | 25 | 100.0% | 8/8 | 8/8 | 8/8 | 1/1 | 0 | 29.1 | 64.6 |
| trim_start_3s | Trim | 23 | 78.3% | 4/8 | 7/8 | 6/6 | 1/1 | 1 | 29.8 | 66.7 |
| trim_start_10s | Trim | 9 | 88.9% | - | 7/8 | - | 1/1 | 1 | 32.6 | 49.2 |
| trim_end_3s | Trim | 23 | 95.7% | 7/8 | 8/8 | 6/6 | 1/1 | 0 | 23.2 | 52.8 |
| trim_end_10s | Trim | 9 | 100.0% | - | 8/8 | - | 1/1 | 0 | 31.6 | 56.8 |
| timeshift_cut_5s | Time shift | 11 | 100.0% | - | 8/8 | 2/2 | 1/1 | 0 | 31.2 | 47.6 |
| timeshift_cut_20s | Time shift | 9 | 88.9% | - | 7/8 | - | 1/1 | 0 | 51.7 | 151.6 |
| timeshift_pad_5s | Time shift | 25 | 100.0% | 8/8 | 8/8 | 8/8 | 1/1 | 0 | 27.8 | 47.3 |
| timeshift_pad_20s | Time shift | 25 | 100.0% | 8/8 | 8/8 | 8/8 | 1/1 | 0 | 31.9 | 45.1 |

Overall edited accuracy (all modified conditions pooled): 618/684 = 90.3%  | latency mean 77.3 ms, p95 229.1 ms

| Negatives (not in DB) | n | false positives | FPR |
|---|---:|---:|---:|
| librispeech_test_clean | 8 | 0 | 0.0% |
| local_music_mp3_segment | 8 | 0 | 0.0% |
| local_speech_mp3_segment | 4 | 0 | 0.0% |
| esc50_time_reversed | 8 | 0 | 0.0% |
| **all** | 28 | 0 | 0.0% |
