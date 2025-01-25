from audiolm_pytorch import FineTransformer, FineTransformerTrainer,SoundStream

# Define the optional checkpoint path
checkpoint_path =""
checkpoint_path = "/home/kaptmoney/Desktop/Audio_Old/soundstream.499.pt"

# Load pretrained SoundStream
soundstream = SoundStream.init_and_load_from(r'/home/kaptmoney/Desktop/Audio_Old/results/fine.transformer.1000.pt')

# Initialize the Fine Transformer
fine_transformer = FineTransformer(
    num_coarse_quantizers=3,
    num_fine_quantizers=5,  # Number of fine quantizers in SoundStream
    codebook_size=1024,  # Adjust based on SoundStream codebook size
    dim=512,
    depth=6,
    flash_attn=True
).cuda()

# Set up the trainer
trainer = FineTransformerTrainer(
    transformer=fine_transformer,
    codec=soundstream,
    folder="/home/kaptmoney/Desktop/Audio_Old/Data//Separate Files",
    batch_size=4,
    grad_accum_every=8,
    data_max_length=320 * 32,
    num_train_steps=1_000_000
)

# Check if checkpoint exists
if checkpoint_path and os.path.exists(checkpoint_path):
    print(f"Checkpoint found. Loading checkpoint from {checkpoint_path}")
    trainer.load(checkpoint_path)
else:
    print("No checkpoint provided or checkpoint file not found. Starting training from scratch.")

# Train the model
trainer.train()
