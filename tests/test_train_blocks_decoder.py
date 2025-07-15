import torch
from src.faith.train.blocks import DecoderBlock, BlockBasedDecoder,EncoderBlock


# Example usage and testing
# Test DecoderBlock
print("Testing DecoderBlock...")
decoder_block = DecoderBlock(
    in_channels=128,
    out_channels=64,
    upsample_factor=(1, 2),
    dropout=0.3,
    upsampling_mode='nearest',
    activation='relu'
)

x = torch.randn(1, 128, 16, 8)
output = decoder_block(x)
print(f"DecoderBlock - Input: {x.shape}, Output: {output.shape}")

# Test configuration
config = decoder_block.get_config()
new_block = DecoderBlock.from_config(config)
print(f"Config serialization successful: {new_block}")

# Test BlockBasedDecoder with mock encoder blocks
print("\nTesting BlockBasedDecoder...")

# Create mock encoder blocks for testing

mock_encoder_blocks = [
    EncoderBlock(80, 128, pool_size=(1, 2)),
    EncoderBlock(128, 256, pool_size=(1, 4)),
    EncoderBlock(256, 128, pool_size=(1, 2)),
]

decoder = BlockBasedDecoder(
    output_channels=80,
    encoder_blocks=mock_encoder_blocks,
    bottleneck_channels=64,
    upsampling_mode='nearest'
)

# Test forward pass
latent = torch.randn(2, 64, 25, 4)
reconstructed = decoder(latent)
print(f"Decoder - Input: {latent.shape}, Output: {reconstructed.shape}")

# Test feature map extraction
feature_maps = decoder.get_feature_maps(latent)
print(f"Feature maps shapes: {[fm.shape for fm in feature_maps]}")

# Test from_encoder class method
decoder2 = BlockBasedDecoder.from_encoder(
    encoder_blocks=mock_encoder_blocks,
    bottleneck_channels=64,
    output_channels=80,
    upsampling_mode='bilinear'
)
print(f"Decoder from encoder: {decoder2}")
