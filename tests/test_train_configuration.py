from src.faith.train.models import (
    ModelConfig,
    create_autoencoder_from_config,
    create_block_autoencoder,
    create_model_from_config_file,
    get_preset_config,
    list_preset_configs,
    save_model_config,
)

# Example usage and testing
# Test preset configurations
print("Available presets:", list_preset_configs())

# Test creating models from presets
for preset_name in ['default', 'light', 'mae_default']:
    print(f"\nTesting {preset_name} preset:")

    try:
        model = create_block_autoencoder(preset_name, input_channels=80)
        print(f"Created model: {type(model).__name__}")

        if hasattr(model, 'parameter_count'):
            print(f"Parameters: {model.parameter_count:,}")

    except Exception as e:
        print(f"Error creating {preset_name}: {e}")

# Test configuration serialization
print("\nTesting configuration serialization:")

config = get_preset_config('default')
config = config.update(input_channels=80, hidden_dim=16)

# Save and load
config.save('test_config.yaml')
loaded_config = ModelConfig.load('test_config.yaml')

print(f"Original: {config.input_channels}, {config.hidden_dim}")
print(
    f"Loaded: {loaded_config.input_channels}, {loaded_config.hidden_dim}")

# Test model config saving
autoencoder = create_autoencoder_from_config(config)
save_model_config(autoencoder, 'model_config.yaml')

# Load and recreate model
recreated_model = create_model_from_config_file('model_config.yaml')
print(f"Recreated model: {type(recreated_model).__name__}")

# Cleanup
import os

os.remove('test_config.yaml')
os.remove('model_config.yaml')

print("Configuration tests completed successfully!")
