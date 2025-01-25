import torch
import torchaudio
from audiolm_pytorch import (
    AudioLM,
    SoundStream,
    HubertWithKmeans,
    SemanticTransformer,
    CoarseTransformer,
    FineTransformer
)

# Paths to trained checkpoints
semantic_transformer_checkpoint = "/home/kaptmoney/Desktop/Audio_Old/results/semantic.transformer.0.pt"
coarse_transformer_checkpoint = "/home/kaptmoney/Desktop/Audio_Old/results/coarse.transformer.1000.pt"
fine_transformer_checkpoint = "/home/kaptmoney/Desktop/Audio_Old/results/fine.transformer.1000.pt"
soundstream_checkpoint = "/home/kaptmoney/Desktop/Audio_Old/0_sex_finetuned/soundstream.1500.pt"

# Load trained SoundStream
print("Loading SoundStream...")
soundstream = SoundStream.init_and_load_from(soundstream_checkpoint)
soundstream.eval()

# Load pretrained HubertWithKmeans
print("Loading HubertWithKmeans...")
wav2vec = HubertWithKmeans(
    checkpoint_path='/home/kaptmoney/Desktop/Audio/Models/hubert_base_ls960.pt',
    kmeans_path='/home/kaptmoney/Desktop/Audio/Models/hubert_base_ls960_L9_km500.bin'
)

# Load trained SemanticTransformer
print("Loading SemanticTransformer...")
semantic_transformer = SemanticTransformer(
    num_semantic_tokens=wav2vec.codebook_size,
    dim=1024,
    depth=6,
    flash_attn=True
)
semantic_transformer.load(semantic_transformer_checkpoint)
semantic_transformer.eval()

# Load trained CoarseTransformer
print("Loading CoarseTransformer...")
coarse_transformer = CoarseTransformer(
    num_semantic_tokens=500,  # Adjust based on kmeans codebook size
    codebook_size=1024,  # Adjust based on SoundStream codebook size
    num_coarse_quantizers=3,  # Number of quantizers in SoundStream
    dim=512,
    depth=6,
    flash_attn=True
)
coarse_transformer.load(coarse_transformer_checkpoint)
coarse_transformer.eval()

# Load trained FineTransformer
print("Loading FineTransformer...")
fine_transformer = FineTransformer(
    num_coarse_quantizers=3,
    num_fine_quantizers=5,  # Number of fine quantizers in SoundStream
    codebook_size=1024,  # Adjust based on SoundStream codebook size
    dim=512,
    depth=6,
    flash_attn=True
)
fine_transformer.load(fine_transformer_checkpoint)

# Initialize AudioLM with loaded components
print("Initializing AudioLM...")
audiolm = AudioLM(
    wav2vec=wav2vec,
    codec=soundstream,
    semantic_transformer=semantic_transformer,
    coarse_transformer=coarse_transformer,
    fine_transformer=fine_transformer
)

# Generate audio using the trained model
print("Generating audio...")
generated_wav = audiolm(batch_size=1)

# Save generated audio to a file
output_path = './generated_audio.wav'
print(f"Saving generated audio to {output_path}...")
torchaudio.save(output_path, generated_wav, sample_rate=24000)

print(f"Audio generated and saved to {output_path}")
