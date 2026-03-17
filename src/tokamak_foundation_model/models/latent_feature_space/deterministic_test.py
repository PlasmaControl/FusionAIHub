import torch
import numpy as np
import matplotlib.pyplot as plt


class DeterministicTestSignals:
    """
    Generate deterministic, interpretable test signals for Perceiver.

    Physics analogy: Simple plasma-like dynamics
    - Signal propagates at constant velocity
    - Actuators modulate amplitude
    - Different modalities show same physics at different rates
    """

    @staticmethod
    def create_test_batch(batch_size=4, d_model=512):
        """
        Create a batch of deterministic test signals.

        Test scenario:
        - Pulse traveling from left to right at constant velocity
        - Fast signals (ts): 10kHz sampling, see detailed motion
        - Slow signals (prof): 100Hz sampling, see coarse motion
        - Video: Spatial pulse moving
        - Actuators: Control pulse amplitude

        Expected Perceiver behavior:
        - Encode: Compress pulse location/amplitude to latent
        - Dynamics: Predict pulse will move right by Δx
        - Decode: Generate pulse at new location
        """

        # Time parameters
        dt_input = 0.5  # 500ms input window
        dt_output = 0.05  # 50ms prediction horizon

        # Pulse parameters (traveling wave)
        pulse_velocity = 1000.0  # samples/second (moves 1000 samples in 1 second)

        signals = {}

        for b in range(batch_size):
            # Each sample has pulse at different starting position
            pulse_start = b * 1000  # Pulse at position 1000, 2000, 3000, 4000

            # Actuator controls amplitude
            actuator_value = 0.5 + 0.5 * (b / batch_size)  # 0.5, 0.625, 0.75, 0.875

            signals[b] = {
                'pulse_start': pulse_start,
                'actuator': actuator_value,
                'velocity': pulse_velocity,
            }

        return signals

    @staticmethod
    def generate_timeseries_tokens(signals, n_tokens=50, d_model=512):
        """
        Generate time series tokens (simulating encoder output).

        Each token represents ~100ms of data (5000 samples / 50 tokens).
        Token should encode: "pulse present in this time window: yes/no, amplitude"
        """
        batch_size = len(signals)
        tokens = torch.zeros(batch_size, n_tokens, d_model)

        for b, sig in signals.items():
            pulse_pos = sig['pulse_start']
            amplitude = sig['actuator']

            # Each token covers ~100 samples (5000 / 50)
            samples_per_token = 5000 / n_tokens

            for token_idx in range(n_tokens):
                token_start = token_idx * samples_per_token
                token_end = (token_idx + 1) * samples_per_token

                # Is pulse in this token's range?
                if token_start <= pulse_pos < token_end:
                    # Encode: "pulse here with this amplitude"
                    tokens[b, token_idx, 0] = 1.0  # Presence flag
                    tokens[b, token_idx, 1] = amplitude  # Amplitude
                    tokens[b, token_idx, 2] = (
                                                          pulse_pos - token_start) / samples_per_token  # Position within token

        return tokens

    @staticmethod
    def generate_profile_tokens(signals, n_tokens=10, d_model=512):
        """
        Generate profile tokens (simulating spatial profile encoder).

        Each token represents a spatial region.
        Profile shows Gaussian peak at pulse location.
        """
        batch_size = len(signals)
        tokens = torch.zeros(batch_size, n_tokens, d_model)

        for b, sig in signals.items():
            # Map pulse position to spatial location (0-50)
            spatial_pos = (sig['pulse_start'] / 5000.0) * 50
            amplitude = sig['actuator']

            # Each token is a spatial region (5 points each)
            for token_idx in range(n_tokens):
                region_center = (token_idx + 0.5) * 5  # Centers at 2.5, 7.5, 12.5, ...

                # Gaussian profile centered at pulse
                distance = abs(region_center - spatial_pos)
                profile_value = amplitude * np.exp(-distance ** 2 / 10.0)

                tokens[b, token_idx, 0] = profile_value  # Profile height
                tokens[b, token_idx, 1] = region_center / 50.0  # Spatial position

        return tokens

    @staticmethod
    def generate_video_tokens(signals, n_tokens=30, d_model=512):
        """
        Generate video tokens (simulating video encoder).

        Video shows bright spot at pulse location moving across frames.
        """
        batch_size = len(signals)
        tokens = torch.zeros(batch_size, n_tokens, d_model)

        for b, sig in signals.items():
            pulse_pos = sig['pulse_start']
            amplitude = sig['actuator']

            # Map to 2D position (256x256 image, 50 frames)
            # Horizontal position based on pulse_pos
            x_pos = (pulse_pos / 5000.0) * 256
            y_pos = 128  # Center vertically

            # Each token represents a spatiotemporal region
            for token_idx in range(n_tokens):
                # Simplified: token encodes if bright spot is in this region
                region_x_start = (token_idx % 6) * 40  # 6 horizontal regions
                region_x_end = region_x_start + 40

                if region_x_start <= x_pos < region_x_end:
                    tokens[b, token_idx, 0] = amplitude  # Brightness
                    tokens[b, token_idx, 1] = (
                                                          x_pos - region_x_start) / 40.0  # Position in region

        return tokens

    @staticmethod
    def generate_expected_output_tokens(signals, dt=0.05, n_tokens_per_modality=None):
        """
        Generate expected output tokens after dynamics.

        Physics: Pulse moves at velocity for dt seconds.
        New position = old position + velocity * dt

        Parameters
        ----------
        signals : dict
            Input signal parameters
        dt : float
            Time step (0.05 seconds = 50ms)
        n_tokens_per_modality : dict
            Number of output tokens per modality
            e.g., {'ts': 50, 'prof': 10, 'vid': 30}

        Returns
        -------
        dict
            Expected output tokens for each modality
        """
        if n_tokens_per_modality is None:
            n_tokens_per_modality = {'ts': 50, 'prof': 10, 'vid': 30}

        batch_size = len(signals)
        d_model = 512

        # Calculate new pulse positions after dt
        new_signals = {}
        for b, sig in signals.items():
            # Pulse moves: new_pos = old_pos + velocity * dt
            displacement = sig['velocity'] * dt  # 1000 * 0.05 = 50 samples
            new_pos = sig['pulse_start'] + displacement

            new_signals[b] = {
                'pulse_start': new_pos,
                'actuator': sig['actuator'],
                'velocity': sig['velocity'],
            }

        # Generate expected tokens for each modality
        expected = {
            'ts': DeterministicTestSignals.generate_timeseries_tokens(
                new_signals, n_tokens_per_modality['ts'], d_model
            ),
            'prof': DeterministicTestSignals.generate_profile_tokens(
                new_signals, n_tokens_per_modality['prof'], d_model
            ),
            'vid': DeterministicTestSignals.generate_video_tokens(
                new_signals, n_tokens_per_modality['vid'], d_model
            ),
        }

        return expected


def test_perceiver_with_deterministic_signals():
    """
    Test Perceiver with deterministic signals and visualize results.

    What the Perceiver should learn:
    1. Encoder: Compress input tokens to latent state
       - Latent should encode: pulse position, amplitude, velocity

    2. Dynamics: Predict future latent state
       - Future position = current position + velocity * dt
       - Amplitude modulated by actuators

    3. Decoder: Expand latent to output tokens
       - Output tokens should show pulse at new position
    """
    from perceiver_components import PerceiverComponents

    # Configuration
    batch_size = 4
    d_model = 512
    n_latent = 256

    # Generate test signals
    print("=== Generating Deterministic Test Signals ===")
    signals = DeterministicTestSignals.create_test_batch(batch_size, d_model)

    for b, sig in signals.items():
        print(f"Sample {b}: pulse_start={sig['pulse_start']}, "
              f"actuator={sig['actuator']:.3f}")

    # Generate input tokens (simulating frozen encoders)
    print("\n=== Generating Input Tokens (Frozen Encoder Output) ===")
    tokens_ts = DeterministicTestSignals.generate_timeseries_tokens(signals, 50, d_model)
    tokens_prof = DeterministicTestSignals.generate_profile_tokens(signals, 10, d_model)
    tokens_vid = DeterministicTestSignals.generate_video_tokens(signals, 30, d_model)

    # Concatenate all input tokens
    all_input_tokens = torch.cat([tokens_ts, tokens_prof, tokens_vid], dim=1)
    print(f"Total input tokens: {all_input_tokens.shape}")  # [4, 90, 512]

    # Extract actuators
    actuators = torch.tensor([sig['actuator'] for sig in signals.values()])
    actuators = actuators.unsqueeze(1).expand(-1, 32)  # [4, 32]

    # Create Perceiver
    print("\n=== Creating Perceiver ===")
    perceiver = PerceiverComponents(
        d_model=d_model,
        n_latent_queries=n_latent,
        n_actuators=32,
        output_queries_config={'ts': 50, 'prof': 10, 'vid': 30},
        encoder_layers=2,
        processor_layers=4,
        decoder_layers=2,
    )

    # Forward pass
    print("\n=== Forward Pass ===")
    output_tokens, latent_current, latent_future = perceiver(
        all_input_tokens,
        actuators
    )

    print(f"Latent current:  {latent_current.shape}")  # [4, 256, 512]
    print(f"Latent future:   {latent_future.shape}")  # [4, 256, 512]
    print(f"Output tokens ts:   {output_tokens['ts'].shape}")  # [4, 50, 512]
    print(f"Output tokens prof: {output_tokens['prof'].shape}")  # [4, 10, 512]
    print(f"Output tokens vid:  {output_tokens['vid'].shape}")  # [4, 30, 512]

    # Generate expected output (what Perceiver should learn to produce)
    print("\n=== Expected Output (After 50ms) ===")
    expected_output = DeterministicTestSignals.generate_expected_output_tokens(
        signals, dt=0.05, n_tokens_per_modality={'ts': 50, 'prof': 10, 'vid': 30}
    )

    for b, sig in signals.items():
        displacement = sig['velocity'] * 0.05
        new_pos = sig['pulse_start'] + displacement
        print(f"Sample {b}: pulse should move from {sig['pulse_start']} "
              f"to {new_pos:.0f} (Δ={displacement})")

    # Visualize
    print("\n=== Visualization ===")
    visualize_perceiver_behavior(
        input_tokens={'ts': tokens_ts, 'prof': tokens_prof, 'vid': tokens_vid},
        output_tokens=output_tokens,
        expected_tokens=expected_output,
        latent_current=latent_current,
        latent_future=latent_future,
        signals=signals
    )


def visualize_perceiver_behavior(
        input_tokens, output_tokens, expected_tokens,
        latent_current, latent_future, signals
):
    """
    Visualize what the Perceiver is doing.
    """
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))

    # Sample to visualize
    sample_idx = 0
    sig = signals[sample_idx]

    # Row 1: Time Series Tokens
    ax = axes[0, 0]
    ax.set_title(f"Input: Time Series Tokens (Sample {sample_idx})")
    ax.imshow(input_tokens['ts'][sample_idx, :, :10].T.detach().numpy(),
              aspect='auto', cmap='viridis')
    ax.set_xlabel('Token Index')
    ax.set_ylabel('First 10 Features')
    ax.axvline(sig['pulse_start'] / 100, color='r', linestyle='--',
               label=f'Pulse at token {sig["pulse_start"] // 100}')
    ax.legend()

    ax = axes[0, 1]
    ax.set_title(f"Output: Time Series Tokens (Expected vs Actual)")
    expected = expected_tokens['ts'][sample_idx, :, 0].detach().numpy()
    actual = output_tokens['ts'][sample_idx, :, 0].detach().numpy()
    ax.plot(expected, 'g-', label='Expected (ground truth)', linewidth=2)
    ax.plot(actual, 'b--', label='Actual (Perceiver output)', linewidth=2)
    new_pos = sig['pulse_start'] + sig['velocity'] * 0.05
    ax.axvline(new_pos / 100, color='r', linestyle='--',
               label=f'Expected pulse at token {new_pos // 100:.0f}')
    ax.legend()
    ax.set_xlabel('Token Index')
    ax.set_ylabel('Feature 0 (Pulse Presence)')

    # Row 2: Profile Tokens
    ax = axes[1, 0]
    ax.set_title(f"Input: Profile Tokens")
    ax.plot(input_tokens['prof'][sample_idx, :, 0].detach().numpy(),
            'o-', label='Profile Value')
    spatial_pos = (sig['pulse_start'] / 5000.0) * 50
    ax.axvline(spatial_pos / 5, color='r', linestyle='--',
               label=f'Pulse at spatial {spatial_pos:.1f}')
    ax.legend()
    ax.set_xlabel('Token Index (Spatial Region)')
    ax.set_ylabel('Profile Height')

    ax = axes[1, 1]
    ax.set_title(f"Output: Profile Tokens (Expected vs Actual)")
    expected = expected_tokens['prof'][sample_idx, :, 0].detach().numpy()
    actual = output_tokens['prof'][sample_idx, :, 0].detach().numpy()
    ax.plot(expected, 'g-', label='Expected', linewidth=2)
    ax.plot(actual, 'b--', label='Actual', linewidth=2)
    ax.legend()
    ax.set_xlabel('Token Index (Spatial Region)')
    ax.set_ylabel('Profile Height')

    # Row 3: Latent Space
    ax = axes[2, 0]
    ax.set_title("Latent Current (First 50 dimensions)")
    ax.imshow(latent_current[sample_idx, :, :50].T.detach().numpy(),
              aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xlabel('Latent Query Index')
    ax.set_ylabel('Dimension')

    ax = axes[2, 1]
    ax.set_title("Latent Future - Latent Current (Change)")
    diff = (latent_future - latent_current)[sample_idx, :, :50].T.detach().numpy()
    im = ax.imshow(diff, aspect='auto', cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    ax.set_xlabel('Latent Query Index')
    ax.set_ylabel('Dimension')
    plt.colorbar(im, ax=ax, label='Change in Latent')

    plt.tight_layout()
    plt.savefig('perceiver_deterministic_test.png', dpi=150)
    print("Saved visualization to: perceiver_deterministic_test.png")
    plt.show()


if __name__ == "__main__":
    test_perceiver_with_deterministic_signals()
