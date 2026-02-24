# MDSPlus Batch Data Fetcher

Automated framework for fetching large-scale MDSPlus data from DIII-D tokamak servers with optional Globus transfer to remote clusters.

## Overview

This framework:

- Fetches MDSPlus data from multiple servers (atlas.gat.com, chiron.gat.com)
- Processes shots in parallel using SLURM job arrays
- Handles thousands of signals per shot via automatic chunking
- Optionally transfers files via Globus and cleans up local storage
- Tracks completion state for resume capability

## File Structure

```
.
├── submit_read_mds_batches.sh    # Main submission script
├── read_mds.sh                   # SLURM worker script
├── config_atlas.yaml             # Signal list for atlas server
├── config_chiron.yaml            # Signal list for chiron server
├── README.md                     # This file
├── .completed_shots              # Auto-generated: completed shots
├── .failed_shots                 # Auto-generated: failed shots
└── jobs/                         # Auto-generated: job logs
```

## Quick Start

### 1. Configure Shot Range or List

Edit `submit_read_mds_batches.sh`:

```bash
# Option A: Process a range of shots
MODE="range"
SHOT_START=200000
SHOT_END=200100

# Option B: Process shots from a file
MODE="list"
SHOT_LIST_FILE="shots_to_process.txt"
```

### 2. Select Configuration

```bash
# Choose which server/signals to fetch
CONFIG_FILE="config_atlas.yaml"   # or config_chiron.yaml
```

### 3. Configure Output

```bash
# Where to save HDF5 files
OUTPUT_DIR="/cscratch/steinerp/database/data"

# Batch settings
BATCH_SIZE=1000          # Shots per batch
MAX_SUBMIT_LIMIT=25      # Max concurrent jobs
```

### 4. Configure Globus (Optional)

Edit `read_mds.sh`:

```bash
# Enable/disable automatic transfer
ENABLE_GLOBUS=true       # Set to false to keep files locally

# Globus endpoints (if enabled)
GLOBUS_SOURCE_ENDPOINT="your-source-id"
GLOBUS_DEST_ENDPOINT="your-dest-id"
GLOBUS_DEST_PATH="/path/on/destination/"
```

### 5. Submit Jobs

**Option A: Run in foreground (blocks terminal)**

```bash
./submit_read_mds_batches.sh
```

**Option B: Run in background with nohup (recommended for long runs)**

```bash
nohup ./submit_read_mds_batches.sh > submission_d3d_mdsplus.log 2>&1 &
```

This will:
- Run in background (terminal can be closed)
- Write all output to `submission_d3d_mdsplus.log`
- Return immediately with process ID

**Monitor background job:**

```bash
# Check if still running
ps aux | grep submit_read_mds_batches.sh

# View progress
tail -f submission_d3d_mdsplus.log

# Check completion
grep "Final Summary" submission_d3d_mdsplus.log
```

## Configuration Files

### Signal Configuration (YAML)

```yaml
trees:
  d3d:
    - \D3D::TOP.MAGNETICS.BPOL_PROBE:BP01
    - \D3D::TOP.MAGNETICS.BPOL_PROBE:BP02
  ptdata:
    - \PTDATA::TOP.RESULTS.ETEMP_PROFILE

server: atlas.gat.com
```

- **trees**: Groups signals by MDSPlus tree
- **signals**: Full MDSPlus paths (one per line)
- **server**: MDSPlus server hostname

### Shot List File

Create `shots_to_process.txt`:

```
# Campaign 2025 shots
200000
200015
200032

# Failed shots to retry
200100
200250
```

- One shot number per line
- Lines starting with `#` are comments
- Empty lines ignored

## Output Structure

```
HDF5_FILE.h5
├── 200000/                    # Shot number
│   ├── d3d/                   # Tree name
│   │   ├── \D3D::TOP.SIGNAL/
│   │   │   ├── data           # Signal values
│   │   │   └── dim0           # Time axis
```

## Features

### Automatic Chunking

Large signal lists are automatically split into chunks (default: 100 signals/chunk) to avoid "Argument list too long" errors.

### State Tracking

- `.completed_shots` - Successfully processed shots (skipped on restart)
- `.failed_shots` - Failed shots for review
- Locked file writes prevent race conditions

### Resume Capability

Rerun `submit_read_mds_batches.sh` to:

- Skip already completed shots
- Retry only failed shots
- Continue interrupted processing

### Globus Transfer

When `ENABLE_GLOBUS=true`:

1. File is transferred to remote cluster
2. Transfer completion is verified
3. Local file is deleted to save space
4. Transfer logged to `globus_transfers.log`

When `ENABLE_GLOBUS=false`:

- Files remain in `OUTPUT_DIR`
- No automatic cleanup

## Monitoring

### Check Progress

```bash
# View current status
tail -f jobs/job_*.out

# Count completed/failed
wc -l .completed_shots .failed_shots

# Check queue
squeue -u $USER
```

### View Logs

```bash
# Latest job output
ls -t jobs/job_*.out | head -1 | xargs cat

# Failed shots
cat .failed_shots
```

## Troubleshooting

### No Shots Processed

**Problem**: `No shots to process (all completed or none in range)`

**Solutions**:

- Check shot range: `SHOT_START` and `SHOT_END`
- Verify shots aren't in `.completed_shots`
- For list mode: check `SHOT_LIST_FILE` exists and contains shots

### Chunk Failures

**Problem**: `Chunk X/Y FAILED`

**Solutions**:

- Check preserved config: `config_SHOT_chunkN_*.yml`
- Verify server connectivity: `ping atlas.gat.com`
- Check signal paths in config file
- Review job logs in `jobs/` directory

### Globus Errors

**Problem**: `Transfer submission failed`

**Solutions**:

- Verify endpoints are activated
- Check endpoint IDs are correct
- Ensure collection paths are accessible
- Re-authenticate: `globus login`
- Grant data access (see Globus setup below)

### Memory Errors

**Problem**: `Out of memory`

**Solutions**:

- Reduce `CHUNK_SIZE` in `read_mds.sh` (default: 100)
- Increase memory: `#SBATCH --mem=128G`
- Process fewer signals per config

## Globus Setup

### One-Time Setup

```bash
# Install Globus CLI
module load mdsplus
pip3 install globus-cli

# Authenticate
globus login

# Grant collection access
globus session consent 'urn:globus:auth:scope:transfer.api.globus.org:all[*https://auth.globus.org/scopes/COLLECTION_ID/data_access]'
```

### Find Endpoint IDs

1. Go to https://app.globus.org/file-manager
2. Select your collection
3. Copy ID from URL: `?origin_id=ENDPOINT_ID`

### Test Transfer

```bash
globus ls ENDPOINT_ID:/path/to/files/
globus transfer SOURCE_ID:/path/file.h5 DEST_ID:/path/file.h5
```

## Advanced Usage

### Process Specific Shots

```bash
# Create shot list
echo -e "200000\n200015\n200032" > my_shots.txt

# Configure
MODE="list"
SHOT_LIST_FILE="my_shots.txt"

# Submit
./submit_read_mds_batches.sh
```

### Retry Failed Shots

```bash
# Use failed shots as input
cp .failed_shots shots_to_retry.txt

# Clear failed list
> .failed_shots

# Configure and submit
MODE="list"
SHOT_LIST_FILE="shots_to_retry.txt"
./submit_read_mds_batches.sh
```

### Multiple Configurations

```bash
# Submit atlas jobs
CONFIG_FILE="config_atlas.yaml"
./submit_read_mds_batches.sh &

# Submit chiron jobs  
CONFIG_FILE="config_chiron.yaml"
./submit_read_mds_batches.sh &
```

## Performance Tips

- **Chunk size**: Smaller = more overhead, larger = higher memory
- **Batch size**: Balance between queue management and parallelism
- **Max jobs**: Respect cluster limits
- **Globus**: Disable if processing locally or transferring later

## Support

For issues:
1. Check job logs: `jobs/job_*.err`
2. Check Globus status: https://app.globus.org/activity
