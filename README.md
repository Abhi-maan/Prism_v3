# Samsung PRISM — Audio Intelligence Platform

A unified web platform integrating three advanced audio analysis models built for the Samsung PRISM Innovation Track. This application wraps three independent research codebases into a single, cohesive, premium web interface using Flask.

## 🌟 Features

This unified platform allows users to upload audio files (WAV, MP3, FLAC, OGG, M4A, AAC, AU) and run them through one of three dedicated intelligence modules:

### 1. 🔄 Duplicate & Edited Audio Detection (SONIC)
Detects whether an audio clip is a direct duplicate or an edited version of a track already present in a local database.
- Uses the **SONIC HNSW neural fingerprint pipeline**.
- **Smart Fallback:** If a pre-built database is not found, the module intelligently runs in "Batch-only Mode", comparing all the files you just uploaded against each other to find duplicates within the batch.

### 2. 🧠 Synthetic AI Detection (PRISM Echoes)
Identifies whether an audio file is AI-generated (e.g., using vocoders, GANs, or diffusion models).
- Uses the **PRISM ensemble classifier**.
- Achieves >90% accuracy across speech, music, and environmental domains.
- Provides deep-dive confidence scores across multiple AI generation techniques.

### 3. 🌐 Internet Audio Search (AcoustID)
Checks if an audio file is available on the public internet.
- Uses **Chromaprint acoustic fingerprinting**.
- Performs a global lookup against the **AcoustID / MusicBrainz** database.
- Returns verified track details and external database links if a match is found.

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- Windows (due to `fpcalc.exe` dependencies for the internet search module)

### 1. Install Dependencies

Install Flask for the web server, plus the core dependencies for the duplicate detection pipeline:

```bash
pip install flask faiss-cpu numpy scipy librosa soundfile pydub psutil matplotlib
```

*(Note: Depending on your environment, you may also need to install dependencies listed in the sub-directories of each module).*

### 2. Prepare the Models

### 2. Reference Database & Models

**Module 1 (Duplicate Detection) - Reference Database & De-duplication:**
You can manage and maintain your reference database directly through the Web UI:
- **No Database Attached:** Click **"+ Add Reference Database"** to recursively scan any folder of audios and convert them into SONIC fingerprints with live progress.
- **Database Operations:**
  1. **Add Over Existing Database (Merge):** Converts new audios and appends them into the existing fingerprint index.
  2. **Replace with New Database:** Wipes older database and builds fresh from the new folder.
- **🛡️ Smart Ingestion De-duplication (Automatic Protection):**
  - **Check Against Existing Library:** When appending new audios, incoming files are checked against already indexed files; any exact duplicates or edited copies are automatically filtered out.
  - **Intra-Batch De-duplication:** Checks within the incoming batch itself so only the first/canonical version of duplicate tracks is indexed, preventing database bloat.
- **🔍 Database Audit & Cleaning:**
  - Click **"🔍 Audit Database"** in the UI to run an acoustic cross-check across all currently indexed tracks.
  - Generates a detailed report identifying internal duplicates or edited copies.
  - One-click **"🗑️ Remove All Duplicates from DB"** automatically purges redundant `.npz` fingerprints, cleans `manifest.json`, and rebuilds the FAISS index.

**Module 2 (Synthetic AI Detection) - Pre-Trained Model & Retraining:**
- **Checking Audio (Inference):** The pre-trained ensemble model (`models/prism_echoes_ensemble.pkl`) is already bundled and ready out of the box — **no dataset is required as a prerequisite for checking audios**.
- **Model Retraining:** If you wish to retrain the ML model on custom or newer datasets, click **"⚙️ Train Model with Newer Dataset"** in the Web UI, specify your dataset path, and the training pipeline will run with real-time progress.


---

## 🚀 Running the Web App

Start the Flask server from the root directory:

```bash
python app.py
```

The application will start on **`http://localhost:5000`**. 

Open this URL in your web browser. You will be greeted by the Samsung PRISM web interface. The interface features a status bar that automatically checks the health of all three modules and displays their readiness.

---

## 📂 Project Structure

```text
d:\Prism\
│
├── app.py                                        # The main Flask web server
├── sonic_db_manager.py                           # Persistent database manager, audit & deduplication engine
├── synthetic_train_manager.py                    # Model training pipeline for synthetic detection
├── templates/
│   └── index.html                                # Master web interface (5-stage workflow)
├── sonic_db_store/                               # Persistent reference database (manifest, fingerprints, FAISS)
├── uploads/                                      # Temporary staging folder for uploaded files
│
├── SONIC-fixed_1/                                # Module 1: SONIC duplicate & edited audio retrieval
├── syntehetic audio detection/                   # Module 2: PRISM AI synthetic detection ensemble
└── internet fast search/                         # Module 3: AcoustID / MusicBrainz availability checker
```

## 🧹 How Uploads Work
The `uploads/` folder is used as a temporary scratchpad. When you upload files via the web UI, `app.py` saves them to a unique session folder inside `uploads/`. Once the chosen module finishes processing the audio, the backend automatically deletes the files and cleans up the folder to preserve disk space.
