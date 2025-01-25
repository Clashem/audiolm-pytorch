from audiolm_pytorch import HubertWithKmeans, SemanticTransformer, SemanticTransformerTrainer

# Load pretrained HubertWithKmeans
wav2vec = HubertWithKmeans(
    checkpoint_path='/home/kaptmoney/Desktop/Audio/Models/hubert_base_ls960.pt',
    kmeans_path='/home/kaptmoney/Desktop/Audio/Models/hubert_base_ls960_L9_km500.bin'
)

# Initialize the Semantic Transformer
semantic_transformer = SemanticTransformer(
    num_semantic_tokens = wav2vec.codebook_size,
    # num_semantic_tokens=500,
    dim=1024,
    depth=6,
    flash_attn=True
).to("cuda")

# Set up the trainer
trainer = SemanticTransformerTrainer(
    transformer=semantic_transformer,
    wav2vec=wav2vec,
    folder="/home/kaptmoney/Desktop/Audio_Old/Data//Separate Files",
    batch_size=4,
    grad_accum_every=8,
    data_max_length=320 * 32,
    num_train_steps=1,
    # save_model_every=1,
)

# Train the model
if __name__ == "__main__":
    trainer.train()
