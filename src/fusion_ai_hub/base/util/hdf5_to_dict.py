def hdf5_to_dict(self, group):
    result = {}
    for key in group.keys():
        if isinstance(group[key], h5py.Dataset):
            result[key] = group[key][()]
        elif isinstance(group[key], h5py.Group):
            result[key] = self.hdf5_to_dict(group[key])
    return result