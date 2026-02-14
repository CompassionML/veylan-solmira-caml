# StrongCompute Quick Connection Guide

**Account:** veylan.solmira@gmail.com
**Container:** veylan-initial-2026-01-03 (50GB, based on NewestCaML)
**Cluster:** Sydney Compute Cluster

---

## Prerequisites (Already Done)

- [x] StrongCompute account registered
- [x] Added to CaML organization
- [x] SSH keys generated and uploaded
- [x] Container created (veylan-initial-2026-01-03)
- [x] Container upgraded to 50GB
- [x] Wireguard VPN configured for Sydney cluster

---

## Quick Start

### 1. Connect to VPN

Start Wireguard and connect to Sydney cluster VPN.

```bash
# Check status anytime
./connect.sh --status
```

### 2. Start Container (Web UI)

**Note:** No API/CLI for starting containers — confirmed via Discord `#isc-help`. Must use web UI.

1. Go to [Control Plane](https://cp.strongcompute.ai/workstations)
2. Find `veylan-initial-2026-01-03`
3. Click **Start**
4. Select:
   - **Cluster:** Sydney (or Burst if needed)
   - **GPUs:** As needed (1 for dev, 8 for training)
5. Wait for status → **Running** (green)
6. Copy the SSH hostname and port shown

### 3. Connect

```bash
./connect.sh <hostname> <port>

# Example:
./connect.sh 192.168.127.170 47180
```

The script will:
- Check VPN connectivity
- Update SSH config
- Test container reachability
- Connect and auto-setup environment (venv, GPU info)

---

## connect.sh Reference

Single script for all connection needs.

### Modes

| Command | Description |
|---------|-------------|
| `./connect.sh <host> <port>` | Connect to container (default) |
| `./connect.sh <host> <port> --vscode` | Open in VSCode Remote SSH |
| `./connect.sh <host> <port> --no-connect` | Just update SSH config |
| `./connect.sh --status` | Check VPN + container status |
| `./connect.sh --help` | Show help |

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SSH_KEY` | Path to SSH private key | `~/Desktop/ai_dev/caml/secure/caml` |
| `SKIP_VPN` | Set to `1` to skip VPN check | `0` |

### Examples

```bash
# Check if everything is ready
./connect.sh --status

# Connect to container
./connect.sh 192.168.127.145 44362

# Open in VSCode instead
./connect.sh 192.168.127.145 44362 --vscode

# Just update config (for later connection)
./connect.sh 192.168.127.145 44362 --no-connect

# Then connect manually anytime
ssh strongcompute
```

---

## ISC CLI Commands

Run these from **inside** the container:

| Command | Description |
|---------|-------------|
| `isc ping` | Verify authentication |
| `isc experiments` | List experiments |
| `isc train /path/to/experiment.isc` | Start training |
| `isc cancel <experiment-id>` | Cancel experiment |
| `isc container restart` | Commit changes + restart |
| `isc container stop` | Commit changes + stop |

### Flags

- `--squash` or `-s` — Reduce image size (recommended)
- `--no-save` — Skip backup (faster, but risky)

---

## Stop Container (Important!)

**Containers cost credits while running. Always stop when done.**

### Option 1: From inside container
```bash
isc container stop --squash
```

### Option 2: From Web UI
1. Go to [Control Plane](https://cp.strongcompute.ai/workstations)
2. Click **Stop** on your container
3. Enable **Save** and **Squash**
4. Click **Stop**

---

## Troubleshooting

### "Connection refused" or timeout
- Is VPN connected? (`./connect.sh --status`)
- Is container running? (Check Control Plane)
- Did IP/port change? (Re-run connect.sh with new values)

### "Permission denied (publickey)"
- Check SSH key is uploaded to Control Plane
- Verify `SSH_KEY` env var or default path

### Container won't start
- Check credits balance in Control Plane
- Try a different cluster or burst option
- Ask in Discord `#isc-help` channel

---

## Resources

- **Control Plane:** https://cp.strongcompute.ai
- **Docs:** https://docs.strongcompute.com
- **Discord:** https://discord.gg/Qp5p9pqH (use `#isc-help`)
- **VPN Setup:** https://docs.strongcompute.com/getting-started/vpn-sydney-cluster-only
