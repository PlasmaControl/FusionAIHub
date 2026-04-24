import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class DummyTokamakDataset(Dataset):
    """
    Dummy dataset with current AND future actuator states.

    Physics model: Traveling pulse/wave with actuator control
    - Actuators at t control amplitude
    - Actuators at t+dt can change (e.g., power ramp)
    """

    def __init__(
            self,
            n_samples=1000,
            dt=0.05,
            pulse_velocity=1000.0,
            d_model=512,
            seed=42
    ):
        self.n_samples = n_samples
        self.dt = dt
        self.pulse_velocity = pulse_velocity
        self.d_model = d_model

        np.random.seed(seed)
        torch.manual_seed(seed)

        self.n_tokens = {
            'ts': 50,
            'prof': 10,
            'vid': 30,
        }

        self._generate_samples()

    def _generate_samples(self):
        """Pre-generate all sample parameters."""
        self.samples = []

        for i in range(self.n_samples):
            # Random pulse parameters
            pulse_start = np.random.uniform(500, 4500)
            amplitude_current = np.random.uniform(0.3, 1.0)

            # Actuators at time t (current)
            actuator_current = amplitude_current + np.random.randn() * 0.05
            actuator_current = np.clip(actuator_current, 0, 1)

            # Actuators at time t+dt (future) - can change!
            # 70% of time stays same, 30% of time changes
            if np.random.rand() < 0.7:
                actuator_future = actuator_current + np.random.randn() * 0.02
            else:
                # Larger change (ramp, step)
                actuator_future = actuator_current + np.random.uniform(-0.3, 0.3)
            actuator_future = np.clip(actuator_future, 0, 1)

            # Amplitude evolution depends on actuators
            # If actuator increases, amplitude increases
            amplitude_future = amplitude_current + (actuator_future - actuator_current) * 0.5
            amplitude_future = np.clip(amplitude_future, 0.3, 1.0)

            # Velocity (small variations)
            velocity = self.pulse_velocity * np.random.uniform(0.9, 1.1)

            # Calculate future position
            displacement = velocity * self.dt
            pulse_future = pulse_start + displacement

            self.samples.append({
                'pulse_start': pulse_start,
                'pulse_future': pulse_future,
                'amplitude_current': amplitude_current,
                'amplitude_future': amplitude_future,
                'actuator_current': actuator_current,
                'actuator_future': actuator_future,
                'velocity': velocity,
            })

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Generate input tokens (current state)
        input_tokens_dict = {
            'ts': self._generate_ts_tokens(
                sample['pulse_start'],
                sample['amplitude_current']
            ),
            'prof': self._generate_prof_tokens(
                sample['pulse_start'],
                sample['amplitude_current']
            ),
            'vid': self._generate_vid_tokens(
                sample['pulse_start'],
                sample['amplitude_current']
            ),
        }

        # Concatenate input tokens
        input_tokens = torch.cat([
            input_tokens_dict['ts'],
            input_tokens_dict['prof'],
            input_tokens_dict['vid'],
        ], dim=0)

        # Generate target tokens (future state with future amplitude!)
        target_tokens = {
            'ts': self._generate_ts_tokens(
                sample['pulse_future'],
                sample['amplitude_future']
            ),
            'prof': self._generate_prof_tokens(
                sample['pulse_future'],
                sample['amplitude_future']
            ),
            'vid': self._generate_vid_tokens(
                sample['pulse_future'],
                sample['amplitude_future']
            ),
        }

        # Actuators (expand to 32 dims)
        actuators_current = torch.ones(32) * sample['actuator_current']
        actuators_future = torch.ones(32) * sample['actuator_future']

        return {
            'input_tokens': input_tokens,
            'actuators_current': actuators_current,
            'actuators_future': actuators_future,
            'target_tokens': target_tokens,
            'metadata': sample,
        }

    def _generate_ts_tokens(self, pulse_pos, amplitude):
        """Generate time series tokens with pulse at position."""
        tokens = torch.zeros(self.n_tokens['ts'], self.d_model)
        samples_per_token = 5000 / self.n_tokens['ts']

        for token_idx in range(self.n_tokens['ts']):
            token_start = token_idx * samples_per_token
            token_end = (token_idx + 1) * samples_per_token

            if token_start <= pulse_pos < token_end:
                tokens[token_idx, 0] = 1.0
                tokens[token_idx, 1] = amplitude
                tokens[token_idx, 2] = (pulse_pos - token_start) / samples_per_token
                tokens[token_idx, 3:10] = amplitude * torch.randn(7) * 0.1

        return tokens

    def _generate_prof_tokens(self, pulse_pos, amplitude):
        """Generate profile tokens with Gaussian centered at pulse."""
        tokens = torch.zeros(self.n_tokens['prof'], self.d_model)
        spatial_pos = (pulse_pos / 5000.0) * 50

        for token_idx in range(self.n_tokens['prof']):
            region_center = (token_idx + 0.5) * 5
            distance = abs(region_center - spatial_pos)
            profile_value = amplitude * np.exp(-distance**2 / 10.0)

            tokens[token_idx, 0] = profile_value
            tokens[token_idx, 1] = region_center / 50.0
            tokens[token_idx, 2:8] = profile_value * torch.randn(6) * 0.05

        return tokens

    def _generate_vid_tokens(self, pulse_pos, amplitude):
        """Generate video tokens with bright spot at pulse location."""
        tokens = torch.zeros(self.n_tokens['vid'], self.d_model)
        x_pos = (pulse_pos / 5000.0) * 256

        n_regions_x = 6
        region_width = 256 / n_regions_x

        for token_idx in range(self.n_tokens['vid']):
            region_idx = token_idx % n_regions_x
            region_x_start = region_idx * region_width
            region_x_end = region_x_start + region_width

            if region_x_start <= x_pos < region_x_end:
                tokens[token_idx, 0] = amplitude
                tokens[token_idx, 1] = (x_pos - region_x_start) / region_width
                tokens[token_idx, 2:12] = amplitude * torch.randn(10) * 0.1

        return tokens


def collate_fn(batch):
    """Collate function for DataLoader."""
    return {
        'input_tokens': torch.stack([item['input_tokens'] for item in batch]),
        'actuators_current': torch.stack([item['actuators_current'] for item in batch]),
        'actuators_future': torch.stack([item['actuators_future'] for item in batch]),
        'target_tokens': {
            'ts': torch.stack([item['target_tokens']['ts'] for item in batch]),
            'prof': torch.stack([item['target_tokens']['prof'] for item in batch]),
            'vid': torch.stack([item['target_tokens']['vid'] for item in batch]),
        },
        'metadata': [item['metadata'] for item in batch],
    }


def create_dummy_dataloaders(
        n_train=8000,
        n_val=1000,
        batch_size=32,
        num_workers=4,
        seed=42
):
    """Create train and validation dataloaders."""
    train_dataset = DummyTokamakDataset(
        n_samples=n_train,
        dt=0.05,
        pulse_velocity=1000.0,
        d_model=512,
        seed=seed
    )

    val_dataset = DummyTokamakDataset(
        n_samples=n_val,
        dt=0.05,
        pulse_velocity=1000.0,
        d_model=512,
        seed=seed + 1
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    return train_loader, val_loader


# Example usage and verification
if __name__ == "__main__":
    print("=== Creating Dummy Dataset ===")

    # Create dataloaders
    train_loader, val_loader = create_dummy_dataloaders(
        n_train=1000,
        n_val=200,
        batch_size=4,
        num_workers=0  # 0 for debugging
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")

    # Inspect a batch
    print("\n=== Inspecting First Batch ===")
    batch = next(iter(train_loader))

    print(f"Input tokens shape:  {batch['input_tokens'].shape}")
    print(f"Actuators shape:     {batch['actuators'].shape}")
    print(f"Target tokens:")
    for modality, tokens in batch['target_tokens'].items():
        print(f"  {modality}: {tokens.shape}")

    # Verify pulse movement
    print("\n=== Verifying Pulse Dynamics ===")
    for i in range(4):
        meta = batch['metadata'][i]
        print(f"Sample {i}:")
        print(f"  Start pos: {meta['pulse_start']:.1f}")
        print(f"  End pos:   {meta['pulse_future']:.1f}")
        print(f"  Displacement: {meta['pulse_future'] - meta['pulse_start']:.1f}")
        print(f"  Amplitude: {meta['amplitude']:.3f}")
        print(f"  Velocity:  {meta['velocity']:.1f}")

    # Verify token structure
    print("\n=== Verifying Token Structure ===")
    sample_idx = 0

    # Find where pulse is in input
    ts_input = batch['input_tokens'][sample_idx, :50, :]  # First 50 are ts tokens
    pulse_present = ts_input[:, 0]  # Presence flag
    pulse_token_input = torch.argmax(pulse_present).item()

    # Find where pulse is in target
    ts_target = batch['target_tokens']['ts'][sample_idx, :, :]
    pulse_present_target = ts_target[:, 0]
    pulse_token_target = torch.argmax(pulse_present_target).item()

    print(f"Sample {sample_idx}:")
    print(f"  Input pulse at token:  {pulse_token_input}")
    print(f"  Target pulse at token: {pulse_token_target}")
    print(f"  Token shift: {pulse_token_target - pulse_token_input} "
          f"(expected: ~{50 / 100:.0f} = 0-1 token)")

    # Visualize
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for i in range(min(3, batch['input_tokens'].shape[0])):
        # Input tokens
        ax = axes[0, i]
        ts_in = batch['input_tokens'][i, :50, 0].numpy()
        ax.plot(ts_in, 'b-', label='Input')
        ax.set_title(f'Sample {i}: Input TS Tokens')
        ax.set_xlabel('Token Index')
        ax.set_ylabel('Pulse Presence')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Target tokens
        ax = axes[1, i]
        ts_out = batch['target_tokens']['ts'][i, :, 0].numpy()
        ax.plot(ts_out, 'g-', label='Target')
        ax.set_title(f'Sample {i}: Target TS Tokens')
        ax.set_xlabel('Token Index')
        ax.set_ylabel('Pulse Presence')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Mark expected displacement
        meta = batch['metadata'][i]
        displacement_tokens = (meta['pulse_future'] - meta['pulse_start']) / 100
        ax.text(0.5, 0.9, f"Δ = {displacement_tokens:.1f} tokens",
                transform=ax.transAxes, ha='center')

    plt.tight_layout()
    plt.savefig('dummy_dataset_verification.png', dpi=150)
    print("\nSaved verification plot to: dummy_dataset_verification.png")
    plt.show()
