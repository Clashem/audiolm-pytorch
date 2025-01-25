# Code based upon lucasnewman
import torchaudio
import torch
import os
from audiolm_pytorch import SoundStream, SoundStreamTrainer

# Define the optional checkpoint path
checkpoint_path =""
# checkpoint_path = "/home/kaptmoney/Desktop/Audio_Old/0_sex_finetuned/soundstream.1500.pt"
# checkpoint_path = "/home/kaptmoney/Desktop/Audio_Old/0_train_scrath/soundstream.4000.pt"
checkpoint_path = "/home/kaptmoney/Desktop/Audio_Old/soundstream.499.pt"

# Initialize SoundStream model
soundstream = SoundStream(
    target_sample_hz = 24000,
    channels = 32,
    strides=(3, 4, 5, 8),
    rq_num_quantizers = 8,
    rq_groups = 2,
    # multi_spectral_recon_loss_weight = 1,  # 1 when generator only
    # adversarial_loss_weight = 0,   # 0 when generator only
    # feature_loss_weight = 0,  # 0 when generator only
    multi_spectral_recon_loss_weight = 1e-2,  # 1 when generator only
    adversarial_loss_weight = 1,   # 0 when generator only
    feature_loss_weight = 100,  # 0 when generator only
)

# Initialize trainer
trainer = SoundStreamTrainer(
    soundstream,
    folder="/home/kaptmoney/Desktop/Audio_Old/Data/LibriSpeech",
    # folder="/home/kaptmoney/Desktop/Audio_Old/Data/Separate Files",
    batch_size=24,
    grad_accum_every=5,         # Effective batch size of 130
    data_max_length_seconds=1,  # Train on 1-second audio
    num_train_steps=501,
    dl_num_workers = 15,
    save_model_every=1,
    save_results_every=100,
    # lr=2e-4                   # Learning rate
).cuda()

# Check if checkpoint exists
if checkpoint_path and os.path.exists(checkpoint_path):
    print(f"Checkpoint found. Loading checkpoint from {checkpoint_path}")
    trainer.load(checkpoint_path)
else:
    print("No checkpoint provided or checkpoint file not found. Starting training from scratch.")

# Start training
trainer.train()

# Set SoundStream to evaluation mode
soundstream.eval()

# # Generate random noise for 10 seconds at 16 kHz
# noise = torch.randn(16000 * 10).cuda()  # Shape: (1, samples)
#
# # Generate audio using SoundStream
# generated_audio = soundstream(noise, return_recons_only=True)  # Shape: (1, samples)
#
# # Process the generated audio
# recons = generated_audio.squeeze(0).detach().cpu()  # Remove batch dimension and move to CPU
# recons = recons.unsqueeze(0)  # Add channel dimension -> Shape: (1, samples)
# recons = torch.clamp(recons, -1.0, 1.0)  # Clamp values to valid audio range
#
# # Save the generated audio
# output_path = "generated_audio.wav"
# torchaudio.save(output_path, recons, sample_rate=16000)
# print(f"Audio file saved successfully at: {output_path}")

# # after a lot of training, you can test the autoencoding as so
#
# soundstream.eval() # your soundstream must be in eval mode, to avoid having the residual dropout of the residual VQ necessary for training
#
# audio = torch.randn(10080).cuda()
# recons = soundstream(audio, return_recons_only = True) # (1, 10080) - 1 channel
# output_audio_path = "/home/kaptmoney/Desktop/Audio_Old/results/Test.wav"
#
# # Reshape to match torchaudio's expected input format
# # Assuming the audio tensor is shaped (1, n_samples)
# recons = recons.squeeze(0).detach().cpu()  # Remove batch dimension and move to CPU
# # recons = torch.clamp(recons, -1.0, 1.0)    # Ensure valid range for audio
#
# # Save the audio
# try:
#     torchaudio.save(output_audio_path, recons.unsqueeze(0), sample_rate=24000)
#     print(f"Reconstructed audio saved to {output_audio_path}")
# except Exception as e:
#     print(f"Error saving audio: {e}")
