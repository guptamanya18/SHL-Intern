import os
import torch
import pandas as pd
from tqdm import tqdm
from pre_processing import preprocesor


class MGSNetDatasetPreparer:
    def __init__(self, audio_dir, transcript_dir, output_dir, max_len=128):
        self.audio_dir = audio_dir
        self.label_csv = transcript_dir
        self.output_dir = output_dir
        self.dataset_dir = os.path.join(output_dir, 'samples')
        os.makedirs(self.dataset_dir, exist_ok=True)

    def create_dataset(self):
        # 1. Load filename-label CSV
        df = pd.read_csv(self.label_csv)
        dataset = []
        csv_data = []

        print("[INFO] Starting dataset creation...")

        for _, row in tqdm(df.iterrows(), total=len(df)):
            filename = row['filename']
            label = None

            audio_path = os.path.join(self.audio_dir, filename)
            if not os.path.exists(audio_path):
                print(f"[WARNING] Audio file not found: {filename}")
                continue

            try:
                # 2. Load & normalize audio
                waveform_np, sr = preprocesor.load_audio(audio_path)
                waveform = torch.tensor(waveform_np, dtype=torch.float).unsqueeze(0)

                # 3. Generate Mel-spectrogram
                spec_np = preprocesor.audio_to_melspectrogram(waveform_np, sr)
                spec = torch.tensor(spec_np, dtype=torch.float)
                if spec.dim() == 2:
                    spec = spec.unsqueeze(0)

                # 4. Transcription + Tokenization
                transcript_text = preprocesor.transcribe_audio_with_whisper(audio_path)
                token = preprocesor.tokenize_text(transcript_text)

                # 5. ➕ Text Embedding
                embedding_tensor = preprocesor.get_text_embedding(transcript_text)

                # 6. ➕ Grammar Error Vector & Score
                grammar_info = preprocesor.get_grammar_features(transcript_text)
                grammar_vector = grammar_info['error_vector']
                grammar_score = grammar_info.get('score', torch.tensor(0.0))

                # 7. Build sample dict
                sample = {
                    'audio': waveform,
                    'spectrogram': spec,
                    'text': token,
                    'label': torch.tensor(label, dtype=torch.float),
                    'embedding': embedding_tensor,
                    'grammar_errors': grammar_vector,
                    'grammar_score': grammar_score
                }
                dataset.append(sample)

                # 8. Save each sample in <sample_dir>
                base_name = os.path.splitext(filename)[0]
                sample_dir = os.path.join(self.dataset_dir, base_name)
                os.makedirs(sample_dir, exist_ok=True)

                torch.save(waveform, os.path.join(sample_dir, 'waveform.pt'))
                torch.save(spec, os.path.join(sample_dir, 'spectrogram.pt'))
                torch.save(token, os.path.join(sample_dir, 'tokens.pt'))
                torch.save(embedding_tensor, os.path.join(sample_dir, 'embedding.pt'))
                torch.save(grammar_vector, os.path.join(sample_dir, 'grammar_vector.pt'))
                torch.save(grammar_score, os.path.join(sample_dir, 'grammar_score.pt'))

                with open(os.path.join(sample_dir, 'transcript.txt'), 'w', encoding='utf-8') as f:
                    f.write(transcript_text.strip())

                # 9. Record metadata for CSV
                csv_data.append({
                    'filename': base_name,
                    'label': label,
                    'transcript': transcript_text,
                    'grammar_score': grammar_score.item()
                })

            except Exception as e:
                print(f"[ERROR] Processing {filename}: {e}")
                continue

        # 10. Save full dataset
        torch.save(dataset, os.path.join(self.output_dir, 'mgsnet_dataset.pt'))

        # 11. Save metadata CSV
        pd.DataFrame(csv_data).to_csv(
            os.path.join(self.output_dir, 'mgsnet_metadata.csv'),
            index=False
        )

        print("\n✅ [DONE] Dataset creation complete.")
        print(f"  • PyTorch dataset: {self.output_dir}/mgsnet_dataset.pt")
        print(f"  • Metadata CSV:    {self.output_dir}/mgsnet_metadata.csv")
        print(f"  • Sample folders:  {self.dataset_dir}/")
