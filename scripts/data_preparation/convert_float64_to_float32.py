import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm


def convert_float64_to_float32(src_path, dst_path):
    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        def copy_item(name, obj):
            if isinstance(obj, h5py.Group):
                dst.require_group(name)
            elif isinstance(obj, h5py.Dataset):
                data = obj[:]
                if data.dtype == np.float64:
                    data = data.astype(np.float32)
                dst.create_dataset(name, data=data)

        src.visititems(copy_item)


if __name__ == "__main__":
    for k, filename in enumerate(tqdm(sorted(Path("/scratch/gpfs/EKOLEMEN/foundation_model/").glob("*_processed.h5")))):
        if k <= 5500:
            continue
        src = filename
        dst = filename.with_stem(src.stem + "_2")
        convert_float64_to_float32(src, dst)
        print(f"{src.stat().st_size / 1e9:.2f} GB → {dst.stat().st_size / 1e9:.2f} GB")
        dst.rename(filename)
