# Module 1 robustness benchmark (quick)

- Run: 2026-10-20 04:15:15 | git: 057c0e19+uncommitted-changes | path: app.run_duplicate_detection
- DB: 28409 entries | tracks: 25 ({'user_upload': 1, 'environment': 8, 'music': 8, 'speech': 8})
- Correct = matched AND matched_filename == true DB filename (top-1, reported as Duplicate/Edited Audio).

| Condition | Group | n | Top-1 acc | env | music | speech | user | wrong-match | mean ms | p95 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| duplicate | Duplicate | 25 | 64.0% | 4/8 | 4/8 | 7/8 | 1/1 | 0 | 13.9 | 20.0 |
| pitch_-4st | Pitch shift | 25 | 0.0% | 0/8 | 0/8 | 0/8 | 0/1 | 0 | 14.2 | 17.1 |
| pitch_-2st | Pitch shift | 25 | 0.0% | 0/8 | 0/8 | 0/8 | 0/1 | 0 | 14.2 | 16.4 |
| pitch_-1st | Pitch shift | 25 | 0.0% | 0/8 | 0/8 | 0/8 | 0/1 | 0 | 14.1 | 16.0 |
| pitch_+1st | Pitch shift | 25 | 0.0% | 0/8 | 0/8 | 0/8 | 0/1 | 0 | 14.3 | 16.3 |
| pitch_+2st | Pitch shift | 25 | 0.0% | 0/8 | 0/8 | 0/8 | 0/1 | 0 | 14.4 | 16.7 |
| pitch_+4st | Pitch shift | 25 | 0.0% | 0/8 | 0/8 | 0/8 | 0/1 | 0 | 14.9 | 17.1 |
| speed_-20pct | Speed change | 25 | 0.0% | 0/8 | 0/8 | 0/8 | 0/1 | 0 | 14.3 | 16.5 |
| speed_-10pct | Speed change | 25 | 0.0% | 0/8 | 0/8 | 0/8 | 0/1 | 0 | 14.3 | 15.9 |
| speed_-5pct | Speed change | 25 | 0.0% | 0/8 | 0/8 | 0/8 | 0/1 | 0 | 14.8 | 18.5 |
| speed_+5pct | Speed change | 25 | 0.0% | 0/8 | 0/8 | 0/8 | 0/1 | 0 | 13.7 | 15.4 |
| speed_+10pct | Speed change | 25 | 0.0% | 0/8 | 0/8 | 0/8 | 0/1 | 0 | 13.5 | 15.0 |
| speed_+20pct | Speed change | 25 | 0.0% | 0/8 | 0/8 | 0/8 | 0/1 | 0 | 12.9 | 14.5 |
| mp3_64kbps | Compression | 25 | 44.0% | 2/8 | 3/8 | 6/8 | 0/1 | 0 | 13.7 | 17.0 |
| mp3_128kbps | Compression | 25 | 60.0% | 4/8 | 4/8 | 6/8 | 1/1 | 0 | 13.6 | 17.4 |
| noise_snr10db | Noise | 25 | 32.0% | 1/8 | 2/8 | 5/8 | 0/1 | 0 | 12.1 | 15.2 |
| noise_snr20db | Noise | 25 | 48.0% | 1/8 | 4/8 | 7/8 | 0/1 | 0 | 11.4 | 14.7 |
| gain_-6db | Gain | 25 | 64.0% | 4/8 | 4/8 | 7/8 | 1/1 | 0 | 10.7 | 13.7 |
| gain_-3db | Gain | 25 | 64.0% | 4/8 | 4/8 | 7/8 | 1/1 | 0 | 10.3 | 13.2 |
| gain_+3db | Gain | 25 | 64.0% | 4/8 | 4/8 | 7/8 | 1/1 | 0 | 10.3 | 13.4 |
| gain_+6db | Gain | 25 | 64.0% | 4/8 | 4/8 | 7/8 | 1/1 | 0 | 10.3 | 13.3 |
| highpass_150hz | High-pass | 25 | 44.0% | 3/8 | 1/8 | 7/8 | 0/1 | 0 | 11.7 | 16.5 |
| highpass_200hz | High-pass | 25 | 40.0% | 3/8 | 0/8 | 7/8 | 0/1 | 0 | 11.7 | 15.4 |
| trim_start_3s | Trim | 23 | 0.0% | 0/8 | 0/8 | 0/6 | 0/1 | 0 | 10.8 | 14.3 |
| trim_start_10s | Trim | 9 | 0.0% | - | 0/8 | - | 0/1 | 0 | 14.6 | 19.9 |
| trim_end_3s | Trim | 23 | 47.8% | 0/8 | 4/8 | 6/6 | 1/1 | 0 | 9.2 | 14.1 |
| trim_end_10s | Trim | 9 | 55.6% | - | 4/8 | - | 1/1 | 0 | 12.0 | 18.3 |
| timeshift_cut_5s | Time shift | 15 | 0.0% | - | 0/8 | 0/6 | 0/1 | 0 | 11.9 | 14.1 |
| timeshift_cut_20s | Time shift | 9 | 0.0% | - | 0/8 | - | 0/1 | 0 | 14.6 | 19.8 |
| timeshift_pad_5s | Time shift | 25 | 0.0% | 0/8 | 0/8 | 0/8 | 0/1 | 0 | 7.4 | 7.8 |
| timeshift_pad_20s | Time shift | 25 | 0.0% | 0/8 | 0/8 | 0/8 | 0/1 | 0 | 7.4 | 7.6 |

Overall edited accuracy (all modified conditions pooled): 147/688 = 21.4%  | latency mean 12.5 ms, p95 16.5 ms

| Negatives (not in DB) | n | false positives | FPR |
|---|---:|---:|---:|
| librispeech_test_clean | 8 | 0 | 0.0% |
| local_music_mp3_segment | 8 | 0 | 0.0% |
| local_speech_mp3_segment | 4 | 0 | 0.0% |
| esc50_time_reversed | 8 | 0 | 0.0% |
| **all** | 28 | 0 | 0.0% |
