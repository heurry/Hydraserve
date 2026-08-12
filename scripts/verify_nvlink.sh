#!/bin/bash
# Verify NVLink/PCIe topology for PD separation
# Run: bash scripts/verify_nvlink.sh

set -e

echo "========================================"
echo "HydraServe: Hardware Topology Verification"
echo "========================================"
echo ""

# Check GPU count
echo "[1] GPU Detection"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# Check topology
echo "[2] GPU Topology"
nvidia-smi topo -m
echo ""

# Check P2P capability
echo "[3] Peer-to-Peer Access"
python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.device_count() >= 2:
    can_p2p = torch.cuda.can_device_access_peer(0, 1)
    print(f'P2P GPU 0→1: {can_p2p}')
    print(f'P2P GPU 1→0: {torch.cuda.can_device_access_peer(1, 0)}')
"
echo ""

# NVLink check
echo "[4] NVLink Status"
if nvidia-smi topo -m 2>/dev/null | grep -q "NV[0-9]"; then
    echo "✓ NVLink DETECTED between GPUs"
    nvidia-smi topo -m 2>/dev/null | grep "NV[0-9]" | head -5
    echo "  Expected: NV12 = NVLink with 12 links"
    echo "  Transfer strategy: FULL_TRANSFER (BF16 KV + recurrent states)"
else
    echo "✗ No NVLink detected"
    echo "  GPUs connected via PCIe only"
    echo "  Transfer strategy: QUANTIZED_TRANSFER (INT4 KV + recurrent states)"
    echo "  PCIe P2P bandwidth: 12-16 GB/s → INT4 KV 29ms for 32K"
fi
echo ""

# PCIe bandwidth estimate
echo "[5] PCIe Configuration"
if command -v lspci &>/dev/null; then
    lspci | grep -i "vga\|3d\|nvidia" 2>/dev/null | head -4 || echo "  (lspci not showing NVIDIA devices)"
fi
echo ""

# Summary
echo "========================================"
echo "Summary"
echo "========================================"
echo "HydraServe supports three transfer modes:"
echo "  NVLink (112 GB/s):       FULL_TRANSFER (9ms for 32K)"
echo "  PCIe P2P (12-16 GB/s):   QUANTIZED_TRANSFER (29ms for 32K, INT4 KV)"
echo "  PCIe SHM (8-10 GB/s):    QUANTIZED_TRANSFER (43ms for 32K, fallback)"
echo ""
echo "See docs/ for setup instructions."
