# DeepEP v2 — Pure PCIe Deployment Guide

This guide covers deploying DeepEP v2 on machines with **no NVLink and no RDMA NICs** — using only PCIe interconnect between GPUs. All data transfer and barrier synchronization happen over PCIe via NCCL's LSA (Local Shared Access) domain, with zero network traffic.

## Prerequisites

### Hardware

- NVIDIA H-series GPUs (H100/H200/H800, SM 9.0)
- GPUs connected via PCIe (no NVLink required)
- No RDMA NIC required

**Topology requirement**: GPUs used in a single EP group must be within the same NCCL LSA domain. On typical PCIe machines, GPUs on the same NUMA node form an LSA domain. Check your topology:

```bash
nvidia-smi topo -m
```

- `PXB` / `NODE` / `PHB` connections between GPUs → same LSA domain, EP will work
- `SYS` connections (cross-NUMA) → may not be in the same LSA domain, EP across these GPUs may fail

Example supported topology (all 8 GPUs on NUMA 0):
```
GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6  GPU7
 └─PHB──┘    └─PHB──┘    └─PHB──┘    └─PHB──┘
    └──NODE──┘               └──NODE──┘
         └──────── NUMA 0 ────────┘
```

### Software

| Component       | Tested Version              |
|-----------------|-----------------------------|
| OS              | Ubuntu 22.04                |
| NVIDIA Driver   | ≥ 580.x (Open Kernel)       |
| CUDA Toolkit    | 13.0                        |
| Python          | 3.12                        |
| PyTorch         | 2.12.x (with CUDA 13.0)     |
| NCCL            | 2.30.x (with LSA support)   |

> **Note**: NCCL must support LSA (Local Shared Access). NCCL ≥ 2.30 includes this feature.

## Environment Setup

### 1. Install Conda (if not installed)

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p $HOME/miniforge3
source $HOME/miniforge3/etc/profile.d/conda.sh
```

### 2. Create Conda Environment

```bash
conda create -n deepep python=3.12 -y
conda activate deepep
```

### 3. Install PyTorch

Install PyTorch with CUDA support matching your CUDA toolkit version:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
```

Verify:
```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

### 4. Install NCCL

NCCL ≥ 2.30 with LSA support is required. If your PyTorch ships with a compatible NCCL, you can use it directly. Otherwise, install NCCL from the NVIDIA package repository or build from source.

Set the NCCL path:
```bash
export EP_NCCL_ROOT_DIR=/path/to/nccl
```

If NCCL is bundled with PyTorch, find it:
```bash
python -c "import torch; print(torch.utils.cpp_extension._find_nccl_root())"
```

### 5. Install NVSHMEM

DeepEP's build system requires NVSHMEM headers (for legacy kernels). Install NVSHMEM ≥ 2.11:

```bash
export NVSHMEM_ROOT=/path/to/nvshmem
```

> NVSHMEM is only needed for compilation — it is **not used** in pure PCIe mode at runtime.

## Build & Install

```bash
git clone https://github.com/MengYu10151/DeepEP.git
cd DeepEP
git checkout pcie-no-atomic

pip install -e .
```

The build takes a few minutes. JIT-compiled kernels will be cached on first run.

## Configuration

Pure PCIe mode requires two environment variables:

| Variable              | Value       | Description                                       |
|-----------------------|-------------|---------------------------------------------------|
| `EP_DISABLE_GIN`      | `1`         | Disable RDMA GIN backend (no NIC needed)          |
| `NCCL_LSA_TEAM_SIZE`  | `<EP_SIZE>` | Set NCCL LSA domain size to match EP size         |

Optional debug variable:

| Variable           | Value | Description                               |
|--------------------|-------|-------------------------------------------|
| `EP_BUFFER_DEBUG`  | `1`   | Print initialization debug info           |

## Running Tests

### Verify Correctness (EP=2)

```bash
EP_DISABLE_GIN=1 NCCL_LSA_TEAM_SIZE=2 \
  python tests/elastic/test_ep.py \
    --num-processes 2 --hidden 4096 --num-topk 6 --num-experts 256 \
    --num-tokens 128 --num-sms 8 \
    --allow-hybrid-mode 0
```

Expected output includes lines like:
```
[EP: Rank 0/2] Dispatch diff: xxx (< 0.01)
[EP: Rank 0/2] Combine diff: xxx (< 0.01)
```

The `diff` values should be small (< 0.01), confirming correctness.

### Verify Correctness (EP=4)

```bash
EP_DISABLE_GIN=1 NCCL_LSA_TEAM_SIZE=4 \
  python tests/elastic/test_ep.py \
    --num-processes 4 --hidden 4096 --num-topk 6 --num-experts 256 \
    --num-tokens 128 --num-sms 8 \
    --allow-hybrid-mode 0
```

### Run Performance Benchmark

```bash
for NP in 2 4; do
  for TOKENS in 16 32 64 128 512 1024 2048 4096; do
    echo ">>> EP=$NP TOKENS=$TOKENS"
    EP_DISABLE_GIN=1 NCCL_LSA_TEAM_SIZE=$NP \
      python tests/elastic/test_ep.py \
        --num-processes $NP --hidden 4096 --num-topk 6 --num-experts 256 \
        --num-tokens $TOKENS --num-sms 8 \
        --allow-hybrid-mode 0 --test-first-only \
      | grep "EP:.*0/$NP"
  done
done
```

### Verify Zero NIC Traffic

To confirm all data stays on PCIe and no RDMA traffic is generated:

```bash
# Before the test
NIC=<your_nic_name>  # e.g., enp115s0f0np0, or eth0
RX_BEFORE=$(cat /sys/class/net/$NIC/statistics/rx_bytes)
TX_BEFORE=$(cat /sys/class/net/$NIC/statistics/tx_bytes)

# Run the test
EP_DISABLE_GIN=1 NCCL_LSA_TEAM_SIZE=2 \
  python tests/elastic/test_ep.py \
    --num-processes 2 --hidden 4096 --num-topk 6 --num-experts 256 \
    --num-tokens 128 --num-sms 8 \
    --allow-hybrid-mode 0

# After the test
RX_AFTER=$(cat /sys/class/net/$NIC/statistics/rx_bytes)
TX_AFTER=$(cat /sys/class/net/$NIC/statistics/tx_bytes)
echo "NIC traffic: rx=$((RX_AFTER - RX_BEFORE))B tx=$((TX_AFTER - TX_BEFORE))B"
```

Traffic should be near zero (only background ARP/LLDP noise of a few hundred bytes).

## Performance

Measured on 8× NVIDIA H-series GPUs (PCIe, no NVLink), hidden=4096, topk=6, experts=256, num_sms=8.

### Dispatch Bandwidth (GB/s)

| Tokens | EP=2 | EP=4 |
|--------|------|------|
| 16     | 14   | 15   |
| 32     | 25   | 26   |
| 64     | 40   | 32   |
| 128    | 57   | 35   |
| 512    | 73   | 40   |
| 1024   | 77   | 41   |
| 2048   | 79   | 42   |
| 4096   | 80   | 42   |

### Combine Bandwidth (GB/s)

| Tokens | EP=2 | EP=4 |
|--------|------|------|
| 16     | 21   | 22   |
| 32     | 36   | 31   |
| 64     | 48   | 35   |
| 128    | 60   | 39   |
| 512    | 71   | 41   |
| 1024   | 74   | 41   |
| 2048   | 75   | 41   |
| 4096   | 76   | 41   |

## Technical Details

### How It Works

Standard DeepEP v2 uses NVLink for intra-node communication and RDMA (GIN) for inter-node communication. The barrier synchronization between GPUs relies on PCIe atomic operations (`ptx::red_add_rel_sys`), which are **not supported** on many PCIe topologies.

This branch eliminates both dependencies:

1. **No RDMA**: Setting `EP_DISABLE_GIN=1` skips GIN (RDMA) resource allocation. The NCCL LSA domain is used as the scale-up domain, so all data transfer uses TMA symmetric pointers over PCIe BAR memory — no network traffic.

2. **No PCIe atomics**: The barrier mechanism is redesigned from a shared-counter atomic reduction to per-rank flag stores:
   - Each rank writes a flag (`st_release_sys`, a posted PCIe write) to every peer's memory via LSA symmetric pointers
   - Each rank polls its own local flags (`ld_acquire_sys`) until all peers have arrived
   - `st_release_sys` provides release semantics, guaranteeing prior data writes are visible before the barrier flag — replacing the ordering guarantee from the original atomic operation
   - Single-writer per flag eliminates the need for atomicity

### Limitations

- **EP size ≤ LSA domain size**: The maximum EP size is limited by how many GPUs share an NCCL LSA domain. On machines with GPUs split across NUMA nodes connected by `SYS`, EP may be limited to the GPUs within a single NUMA domain.
- **No scale-out (multi-node)**: Pure PCIe mode is single-node only. There is no inter-node communication path without RDMA.

## Troubleshooting

### "CPU side received count: 0 0 0..."
The kernel is not transferring data. Ensure both `EP_DISABLE_GIN=1` and `NCCL_LSA_TEAM_SIZE=<EP_SIZE>` are set.

### "DeepEP NVLink barrier timeout"
The barrier timed out waiting for a peer rank. Check that all GPUs in the EP group are within the same NCCL LSA domain. Reduce `--num-processes` to use only GPUs on the same NUMA node.

### "NCCL GIN is unavailable"
You forgot to set `EP_DISABLE_GIN=1`.

### Build fails with "No module named 'torch'"
Activate your conda environment before building:
```bash
conda activate deepep
pip install -e .
```
