# 🎓 MGSNet - Multimodal Grammar Scoring Network

> **A Deep Learning-Based Grammar Proficiency Scoring System Using Speech and Text Modalities**

## 📌 Overview

**MGSNet** is an intelligent grammar scoring system that leverages **multimodal inputs** — audio (speech) and its corresponding text — to automatically evaluate grammatical proficiency. It utilizes **mel-spectrograms**, **language embeddings**, **tokenization**, and **grammar error vectors** to assign a holistic **grammar score**.

This project was built during a research hackathon to explore automated language assessment for education and language learning.

## 🚀 Features

- 🎤 **Audio Input Support** — Raw `.wav` files are converted to spectrograms and transcribed using Whisper.
- 🧠 **Multimodal Architecture** — Combines features from both text and audio domains.
- 🔍 **Grammar Analysis** — Computes grammar error vectors and scores based on linguistic rules.
- 📊 **Score Prediction** — Outputs a final numeric score representing the user's grammar proficiency.
- 🧱 **Modular Pipeline** — Clean structure with preprocessing, dataset creation, model training, and evaluation.

## 🧠 Model Pipeline

```
Audio (.wav)  --> Mel-Spectrogram
              |
              --> Whisper Transcript
                             |
              +--------------+-------------+
              |                            |
         Text Tokenizer             Text Embedding
              |                            |
         Grammar Analyzer (Errors & Score)
              |
              +-------+--------+--------+
                      |        |        |
                  [Waveform] [Spectrogram] [Embeddings]
                      ↓        ↓         ↓
                          [Fusion Network]       --> [Score Regression Head]
```

## 📁 Folder Structure

```
MGSNet/
├── dataset_creation/
│   ├── data_convert.py              # Test dataset generator
│   ├── dataset_create.py            # Training dataset creation
│   └── __init__.py
├── model/
│   └── mgsnet_model.py              # MGSNet architecture
├── pre_processing/
│   └── preprocesor.py               # Audio & text preprocessing
├── train/
│   └── train_model.py               # Training pipeline
├── test/
│   └── test_model.py                # Evaluation on test dataset
├── saved_models/
│   └── best_model.pt                # Trained model checkpoint
├── README.md
└── requirements.txt
```

## ⚙️ Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Key Libraries:
- `torch`
- `torchaudio`
- `transformers`
- `openai-whisper`
- `pandas`
- `tqdm`

## 🧪 Dataset Preparation

### 🔹 For Training
Run:
```bash
python dataset_creation/dataset_create.py
```

### 🔹 For Testing
Run:
```bash
python dataset_creation/data_convert.py
```

Make sure you have:
- A folder with `.wav` files
- A `.csv` file with audio file names (column: `filename`)

## 🏋️‍♂️ Training the Model

```bash
python train/train_model.py
```

## 🎯 Evaluating the Model

```bash
python test/test_model.py
```

Outputs predictions and metrics on the test set.

## 📈 Output

- `samples/` folder: Contains processed test samples with all `.pt` tensors
- `test_metadata.csv`: Summary of predictions
- `saved_models/`: Stores best model checkpoint
- Console Logs: Real-time progress and status

## 👥 Team

- **Project Lead**: Manya Gupta
- **Institution**: SHL Hackathon, Chandigarh University

## 📄 License

This project is open-sourced for educational and research purposes.

## 💬 Acknowledgements

- [Whisper by OpenAI](https://github.com/openai/whisper)
- [Hugging Face Transformers](https://huggingface.co)
- [PyTorch](https://pytorch.org)

## 🌟 Future Work

- Add support for more languages
- Integrate with ASR evaluation benchmarks
- Build real-time scoring web app
- Improve grammar scoring using BERT-based grammar classifiers
