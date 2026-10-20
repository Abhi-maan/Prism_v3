#!/usr/bin/env python3
"""
Example: Combining Internet Availability with Duplicate Detection & AI Detection
================================================================================
Demonstrates how to integrate the AcoustID InternetAudioDetector component
into a unified 3-model audio verification pipeline.
"""

from internet_audio_detector import InternetAudioDetector, extract_web_availability_features

def simulate_three_model_pipeline(audio_path: str):
    print(f"\n==================================================")
    print(f"  Analyzing: {audio_path}")
    print(f"==================================================")

    # --------------------------------------------------------------------------
    # MODEL 3: Internet Audio Availability (AcoustID + Chromaprint API)
    # --------------------------------------------------------------------------
    detector = InternetAudioDetector()
    
    # 1. Compute fingerprint once (or reuse if already computed by duplicate model)
    duration, fingerprint = detector.compute_fingerprint(audio_path)
    
    # 2. Look up audio on the internet via AcoustID
    web_res = detector.lookup_fingerprint(fingerprint, duration, file_path=audio_path)

    print(f"[Model 3: Internet Availability]")
    print(f"  - Available Online: {web_res.is_available}")
    print(f"  - Match Confidence: {web_res.confidence:.4f}")
    if web_res.recordings:
        rec = web_res.recordings[0]
        artists = ", ".join([a.get("name", "") if isinstance(a, dict) else str(a) for a in rec.get("artists", [])])
        print(f"  - Identified Track: '{rec.get('title')}' by {artists}")
        print(f"  - MusicBrainz ID:   {rec.get('mbid')}")
    elif web_res.acoustid_id:
        print(f"  - AcoustID UUID:    {web_res.acoustid_id}")
        print(f"  - Web Link:         {web_res.track_url}")
    else:
        print(f"  - Online Status:    No web records found (likely novel or synthetic)")

    # --------------------------------------------------------------------------
    # Extract Feature Vector for downstream classifier
    # --------------------------------------------------------------------------
    features = web_res.to_feature_vector()
    print(f"\n[Feature Vector for Combined Decision Engine]")
    for k, v in features.items():
        print(f"  {k:25s}: {v}")

    # --------------------------------------------------------------------------
    # Decision Logic for 3-Model Fusion
    # --------------------------------------------------------------------------
    print(f"\n[Multi-Model Fusion Interpretation]")
    if web_res.is_available:
        print(f"  -> Audio is a verified web recording (confidence: {web_res.confidence:.2f}).")
        print("     High confidence web match: Excludes purely novel synthetic generation.")
    else:
        print("  -> Audio is unindexed on the internet.")
        print("     Proceed to Model 1 (local duplicate HNSW index) and Model 2 (AI artifact detector).")


if __name__ == "__main__":
    # Test on a verified internet track
    simulate_three_model_pipeline(r"real\archive\genres\pop\pop.00003.au")

    # Test on a synthetic AI audio track
    simulate_three_model_pipeline(r"ai_generated\ATA\audioldm\10000_People_Chanting_Im_an_Individual_Nihilore_audioldm_ATA_001.mp3")
