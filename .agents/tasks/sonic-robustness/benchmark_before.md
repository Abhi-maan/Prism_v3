# Module 1 robustness benchmark (full)

- Run: 2026-10-20 03:48:55 | git: 057c0e19+uncommitted-changes | path: app.run_duplicate_detection
- DB: 28409 entries | tracks: 103 ({'user_upload': 1, 'environment': 34, 'music': 34, 'speech': 34})
- Correct = matched AND matched_filename == true DB filename (top-1, reported as Duplicate/Edited Audio).

| Condition | Group | n | Top-1 acc | env | music | speech | user | wrong-match | mean ms | p95 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| duplicate | Duplicate | 103 | 57.3% | 18/34 | 13/34 | 27/34 | 1/1 | 0 | 12.5 | 18.2 |
| pitch_-4st | Pitch shift | 103 | 0.0% | 0/34 | 0/34 | 0/34 | 0/1 | 0 | 13.4 | 14.9 |
| pitch_-2st | Pitch shift | 103 | 0.0% | 0/34 | 0/34 | 0/34 | 0/1 | 0 | 13.4 | 14.9 |
| pitch_-1st | Pitch shift | 103 | 0.0% | 0/34 | 0/34 | 0/34 | 0/1 | 0 | 13.3 | 14.9 |
| pitch_+1st | Pitch shift | 103 | 0.0% | 0/34 | 0/34 | 0/34 | 0/1 | 0 | 13.3 | 14.8 |
| pitch_+2st | Pitch shift | 103 | 0.0% | 0/34 | 0/34 | 0/34 | 0/1 | 0 | 13.3 | 14.9 |
| pitch_+4st | Pitch shift | 103 | 0.0% | 0/34 | 0/34 | 0/34 | 0/1 | 0 | 13.3 | 15.0 |
| speed_-20pct | Speed change | 103 | 0.0% | 0/34 | 0/34 | 0/34 | 0/1 | 0 | 13.5 | 15.1 |
| speed_-10pct | Speed change | 103 | 0.0% | 0/34 | 0/34 | 0/34 | 0/1 | 0 | 13.4 | 14.8 |
| speed_-5pct | Speed change | 103 | 0.0% | 0/34 | 0/34 | 0/34 | 0/1 | 0 | 13.3 | 14.8 |
| speed_+5pct | Speed change | 103 | 0.0% | 0/34 | 0/34 | 0/34 | 0/1 | 0 | 13.0 | 14.6 |
| speed_+10pct | Speed change | 103 | 0.0% | 0/34 | 0/34 | 0/34 | 0/1 | 0 | 12.8 | 14.4 |
| speed_+20pct | Speed change | 103 | 0.0% | 0/34 | 0/34 | 0/34 | 0/1 | 0 | 12.4 | 14.7 |
| mp3_64kbps | Compression | 103 | 49.5% | 12/34 | 11/34 | 28/34 | 0/1 | 0 | 13.2 | 16.8 |
| mp3_128kbps | Compression | 103 | 57.3% | 17/34 | 13/34 | 28/34 | 1/1 | 0 | 13.1 | 17.1 |
| noise_snr10db | Noise | 103 | 25.2% | 4/34 | 7/34 | 15/34 | 0/1 | 0 | 12.1 | 14.9 |
| noise_snr20db | Noise | 103 | 50.5% | 14/34 | 12/34 | 26/34 | 0/1 | 0 | 10.8 | 14.4 |
| gain_-6db | Gain | 103 | 56.3% | 17/34 | 13/34 | 27/34 | 1/1 | 0 | 10.3 | 14.1 |
| gain_-3db | Gain | 103 | 56.3% | 17/34 | 13/34 | 27/34 | 1/1 | 0 | 10.0 | 13.8 |
| gain_+3db | Gain | 103 | 56.3% | 17/34 | 13/34 | 27/34 | 1/1 | 0 | 10.0 | 13.6 |
| gain_+6db | Gain | 103 | 56.3% | 17/34 | 13/34 | 27/34 | 1/1 | 0 | 9.9 | 13.8 |
| highpass_150hz | High-pass | 103 | 38.8% | 10/34 | 2/34 | 28/34 | 0/1 | 0 | 11.1 | 14.4 |
| highpass_200hz | High-pass | 103 | 34.0% | 9/34 | 0/34 | 26/34 | 0/1 | 0 | 11.3 | 14.5 |
| trim_start_3s | Trim | 92 | 0.0% | 0/34 | 0/34 | 0/23 | 0/1 | 0 | 10.1 | 14.0 |
| trim_start_10s | Trim | 35 | 5.7% | - | 2/34 | - | 0/1 | 0 | 13.5 | 14.5 |
| trim_end_3s | Trim | 92 | 37.0% | 2/34 | 13/34 | 18/23 | 1/1 | 0 | 8.6 | 13.7 |
| trim_end_10s | Trim | 35 | 40.0% | - | 13/34 | - | 1/1 | 0 | 11.4 | 13.7 |
| timeshift_cut_5s | Time shift | 48 | 2.1% | - | 1/34 | 0/13 | 0/1 | 0 | 12.4 | 14.5 |
| timeshift_cut_20s | Time shift | 35 | 0.0% | - | 0/34 | - | 0/1 | 0 | 13.7 | 16.4 |
| timeshift_pad_5s | Time shift | 103 | 0.0% | 0/34 | 0/34 | 0/34 | 0/1 | 0 | 7.4 | 7.7 |
| timeshift_pad_20s | Time shift | 103 | 0.0% | 0/34 | 0/34 | 0/34 | 0/1 | 0 | 7.3 | 7.6 |

Overall edited accuracy (all modified conditions pooled): 546/2809 = 19.4%  | latency mean 11.9 ms, p95 15.5 ms

| Negatives (not in DB) | n | false positives | FPR |
|---|---:|---:|---:|
| librispeech_test_clean | 40 | 0 | 0.0% |
| local_music_mp3_segment | 18 | 0 | 0.0% |
| local_speech_mp3_segment | 20 | 0 | 0.0% |
| esc50_time_reversed | 40 | 0 | 0.0% |
| **all** | 118 | 0 | 0.0% |
