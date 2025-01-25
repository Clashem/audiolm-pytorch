from audiolm_pytorch import HubertWithKmeans,CoarseTransformer, CoarseTransformerTrainer, SoundStream
import os

# Define the optional checkpoint path
checkpoint_path =""
checkpoint_path = "/home/kaptmoney/Desktop/Audio_Old/results/coarse.transformer.1000.pt"

# Load pretrained SoundStream and HubertWithKmeans
soundstream = SoundStream.init_and_load_from(r'/home/kaptmoney/Desktop/Audio_Old/0_sex_finetuned/soundstream.1500.pt')
wav2vec = HubertWithKmeans(
    checkpoint_path='/home/kaptmoney/Desktop/Audio/Models/hubert_base_ls960.pt',
    kmeans_path='/home/kaptmoney/Desktop/Audio/Models/hubert_base_ls960_L9_km500.bin'
)

# Initialize the Coarse Transformer
coarse_transformer = CoarseTransformer(
    num_semantic_tokens=500,  # Adjust based on kmeans codebook size
    codebook_size=1024,  # Adjust based on SoundStream codebook size
    num_coarse_quantizers=3,  # Number of quantizers in SoundStream
    dim=512,
    depth=6,
    flash_attn=True
).cuda()

# Set up the trainer
trainer = CoarseTransformerTrainer(
    transformer=coarse_transformer,
    codec=soundstream,
    wav2vec=wav2vec,
    folder="/home/kaptmoney/Desktop/Audio_Old/Data//Separate Files",
    batch_size=4,
    grad_accum_every=8,
    data_max_length=320 * 32,
    num_train_steps=1002
    save_model_every=1,
    save_results_every=100,
)

# Check if checkpoint exists
if checkpoint_path and os.path.exists(checkpoint_path):
    print(f"Checkpoint found. Loading checkpoint from {checkpoint_path}")
    trainer.load(checkpoint_path)
else:
    print("No checkpoint provided or checkpoint file not found. Starting training from scratch.")

# Train the model
trainer.train()
