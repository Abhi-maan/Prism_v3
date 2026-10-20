# Module 1 robustness benchmark (custom)

- Run: 2026-10-20 06:38:46 | git: 057c0e19+uncommitted-changes | path: app.run_duplicate_detection
- DB: 28409 entries | tracks: 103 ({'user_upload': 1, 'environment': 34, 'music': 34, 'speech': 34})
- Correct = matched AND matched_filename == true DB filename (top-1, reported as Duplicate/Edited Audio).

| Condition | Group | n | Top-1 acc | env | music | speech | user | wrong-match | mean ms | p95 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| duplicate | Duplicate | 103 | 99.0% | 33/34 | 34/34 | 34/34 | 1/1 | 0 | 21.3 | 32.0 |
| varispeed_-20pct | Varispeed (extra) | 103 | 50.5% | 20/34 | 25/34 | 6/34 | 1/1 | 1 | 198.1 | 244.1 |
| varispeed_-10pct | Varispeed (extra) | 103 | 88.3% | 28/34 | 31/34 | 31/34 | 1/1 | 2 | 143.2 | 226.4 |
| varispeed_-5pct | Varispeed (extra) | 103 | 94.2% | 31/34 | 34/34 | 31/34 | 1/1 | 1 | 76.9 | 98.2 |
| varispeed_+5pct | Varispeed (extra) | 103 | 94.2% | 31/34 | 34/34 | 31/34 | 1/1 | 0 | 65.2 | 87.1 |
| varispeed_+10pct | Varispeed (extra) | 103 | 85.4% | 27/34 | 30/34 | 30/34 | 1/1 | 0 | 162.5 | 241.5 |
| varispeed_+20pct | Varispeed (extra) | 103 | 53.4% | 19/34 | 24/34 | 11/34 | 1/1 | 0 | 165.5 | 247.0 |

Extra (not pooled): varispeed_+10pct, varispeed_+20pct, varispeed_+5pct, varispeed_-10pct, varispeed_-20pct, varispeed_-5pct: 480/618 = 77.7%

Overall edited accuracy (all modified conditions pooled): 0/0 = 0.0%  | latency mean 119.1 ms, p95 231.8 ms

| Negatives (not in DB) | n | false positives | FPR |
|---|---:|---:|---:|
| librispeech_test_clean | 1 | 0 | 0.0% |
| local_music_mp3_segment | 1 | 0 | 0.0% |
| local_speech_mp3_segment | 1 | 0 | 0.0% |
| esc50_time_reversed | 1 | 0 | 0.0% |
| **all** | 4 | 0 | 0.0% |
