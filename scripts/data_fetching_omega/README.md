# Globus File Transfer Setup

Automatic file transfer using Globus between Omega and Stellar clusters.

## One-Time Setup

### 1. Install Globus CLI

```bash
module load mdsplus
pip3 install --user globus-cli
```

### 2. Authenticate

```bash
globus login
```

Follow the URL, authenticate with your institution, and paste the authorization code back.

### 3. Grant Collection Access

Run for **both** source and destination collections:

```bash
globus session consent 'urn:globus:auth:scope:transfer.api.globus.org:all[*https://auth.globus.org/scopes/COLLECTION_ID/data_access]'
```

Replace `COLLECTION_ID` with:
- Omega collection ID: `20749357-d221-43c6-bbc4-79691e6776b8`
- Stellar collection ID: `544b12dc-cb3d-11e9-939b-02ff96a5aa76`

Or simply run `globus session update` and grant access when prompted.

## Configuration

### Find Collection IDs

1. Go to https://app.globus.org/file-manager
2. Search for your collection
3. Copy the ID from the URL: `?origin_id=COLLECTION_ID`

### Minimal Working Example

```bash
#!/bin/bash

module load mdsplus

# Globus configuration
GLOBUS_SOURCE_ENDPOINT="20749357-d221-43c6-bbc4-79691e6776b8"  # Omega
GLOBUS_DEST_ENDPOINT="544b12dc-cb3d-11e9-939b-02ff96a5aa76"    # Stellar
GLOBUS_DEST_PATH="/scratch/gpfs/EKOLEMEN/big_d3d_data/"

# Example file to transfer
OUTPUT_FILE="/cscratch/steinerp/database/data/example.h5"
OUTPUT_FILENAME=$(basename "${OUTPUT_FILE}")

# Strip /cscratch/ mount point (Omega-specific)
GLOBUS_SOURCE_PATH="${OUTPUT_FILE#/cscratch/}"

# Transfer
TRANSFER_TASK_ID=$(globus transfer \
    --preserve-mtime \
    --label "Transfer ${OUTPUT_FILENAME}" \
    --jmespath 'task_id' \
    --format unix \
    "${GLOBUS_SOURCE_ENDPOINT}:${GLOBUS_SOURCE_PATH}" \
    "${GLOBUS_DEST_ENDPOINT}:${GLOBUS_DEST_PATH}${OUTPUT_FILENAME}")

echo "Transfer submitted: ${TRANSFER_TASK_ID}"

# Wait for completion
globus task wait "${TRANSFER_TASK_ID}" --timeout 7200 --polling-interval 30

# Delete local file after successful transfer (optional)
if [ $? -eq 0 ]; then
    rm -f "${OUTPUT_FILE}"
    echo "Transfer complete, local file deleted"
fi
```

## Important: Omega Mount Point

The Omega Globus collection is mounted at `/cscratch/`. Always strip this prefix:

```bash
# If OUTPUT_FILE="/cscratch/steinerp/data/file.h5"
GLOBUS_SOURCE_PATH="${OUTPUT_FILE#/cscratch/}"  # becomes "steinerp/data/file.h5"
```

## Testing

```bash
# Test access to both collections
globus ls 20749357-d221-43c6-bbc4-79691e6776b8:/steinerp/
globus ls 544b12dc-cb3d-11e9-939b-02ff96a5aa76:/scratch/gpfs/EKOLEMEN/

# Test manual transfer
globus transfer \
    20749357-d221-43c6-bbc4-79691e6776b8:steinerp/test.txt \
    544b12dc-cb3d-11e9-939b-02ff96a5aa76:/scratch/gpfs/EKOLEMEN/test.txt
```

## Troubleshooting

**"Missing required data_access consent"**

```bash
globus session update
```

**Check transfer status**

```bash
globus task list
globus task show TASK_ID
```

Or visit: https://app.globus.org/activity

## Resources

- [Globus Documentation](https://docs.globus.org/)
- [Globus CLI Reference](https://docs.globus.org/cli/)
