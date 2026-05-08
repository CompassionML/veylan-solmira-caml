#!/usr/bin/env python3
"""
RunPod GPU Instance Launcher

Launches and manages GPU instances on RunPod with SSH access via direct TCP.

Setup:
  1. Create a .env file (outside the repo, or anywhere in a parent directory):
       RUNPOD_API_KEY=your_api_key_here
       RUNPOD_SSH_PUBLIC_KEY=ssh-ed25519 AAAA... user@host
  2. Generate an SSH key pair if you don't have one:
       ssh-keygen -t ed25519 -f ~/.ssh/runpod_ed25519

Requirements for SSH into pods:
  - Secure Cloud (for public IP)
  - Port 22/tcp exposed
  - PUBLIC_KEY env var with SSH public key
  - dockerArgs to start sshd and keep container running
"""

import os
import requests
import time
import json
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

# Load .env from nearest ancestor directory
load_dotenv(find_dotenv(usecwd=False))

# Configuration
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")
if not RUNPOD_API_KEY:
    print("Error: RUNPOD_API_KEY not set. Create a .env file or set it as an env var.")
    print("See: https://www.runpod.io/console/user/settings -> API Keys")
    sys.exit(1)
API_URL = "https://api.runpod.io/graphql"
REST_URL = "https://rest.runpod.io/v1"

# SSH public key for authentication (loaded from .env or env var)
SSH_PUBLIC_KEY = os.environ.get("RUNPOD_SSH_PUBLIC_KEY", "")

# SSH setup prefix: always runs to enable SSH access regardless of mode
SSH_SETUP = 'mkdir -p ~/.ssh && chmod 700 ~/.ssh && echo \\"$PUBLIC_KEY\\" >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys && service ssh start'


def build_docker_args(cmd: str = None) -> str:
    """Build dockerArgs string. Always starts sshd, then runs cmd or sleeps."""
    if cmd:
        # Job mode: start sshd, run user command, exit when done
        return f'bash -c "{SSH_SETUP} && {cmd}"'
    else:
        # Interactive mode: start sshd, keep alive for SSH access
        return f'bash -c "{SSH_SETUP} && sleep infinity"'


# GPU preset tiers for fine-tuning workloads (use --gpu-type for arbitrary GPU IDs)
GPU_PRESETS = {
    "budget":      "NVIDIA RTX A4000",        # ~$0.25/hr secure, 16GB — QLoRA 8B only
    "value":       "NVIDIA RTX A6000",        # ~$0.49/hr secure, 48GB — LoRA up to 13B
    "recommended": "NVIDIA L40S",             # ~$0.86/hr secure, 48GB — fast LoRA, modern arch
    "powerful":    "NVIDIA A100 80GB PCIe",   # ~$1.39/hr secure, 80GB — large batches, 30B+ QLoRA
}

# Approximate secure-cloud prices ($/hr) for cost estimation when API lookup fails
GPU_PRICE_HINTS = {
    "NVIDIA RTX A4000":       0.25,
    "NVIDIA RTX A6000":       0.49,
    "NVIDIA L40S":            0.86,
    "NVIDIA A100 80GB PCIe":  1.39,
    "NVIDIA A100-SXM4-80GB":  1.49,
    "NVIDIA H100 80GB HBM3":  3.29,
}


def graphql_request(query: str, variables: dict = None) -> dict:
    """Make a GraphQL request to RunPod API."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RUNPOD_API_KEY}"
    }
    payload = {"query": query}
    if variables:
        payload["variables"] = variables

    response = requests.post(API_URL, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()


def rest_request(endpoint: str) -> dict:
    """Make a REST API request to RunPod."""
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}"
    }
    response = requests.get(f"{REST_URL}/{endpoint}", headers=headers)
    response.raise_for_status()
    return response.json()


def check_balance() -> float:
    """Check current RunPod balance."""
    query = "query { myself { clientBalance } }"
    result = graphql_request(query)
    return result["data"]["myself"]["clientBalance"]


def list_available_gpus() -> list:
    """List available GPU types with pricing."""
    query = """
    query {
        gpuTypes {
            id
            displayName
            memoryInGb
            securePrice
            communityPrice
        }
    }
    """
    result = graphql_request(query)
    return result["data"]["gpuTypes"]


def get_gpu_price(gpu_type: str) -> float | None:
    """Look up the secure-cloud price for a GPU type. Returns $/hr or None."""
    # Check local hints first (avoids API call for known presets)
    if gpu_type in GPU_PRICE_HINTS:
        return GPU_PRICE_HINTS[gpu_type]
    # Fall back to API
    try:
        gpus = list_available_gpus()
        for gpu in gpus:
            if gpu["id"] == gpu_type or gpu.get("displayName") == gpu_type:
                return gpu.get("securePrice")
    except Exception:
        pass
    return None


def list_network_volumes() -> list:
    """List existing network volumes via REST API."""
    return rest_request("networkvolumes")


def create_pod(
    gpu_type: str,
    image: str,
    name: str,
    volume_gb: int,
    gpu_count: int,
    network_volume_id: str = None,
    cmd: str = None,
) -> dict:
    """Create a new GPU pod with SSH access."""
    if not SSH_PUBLIC_KEY:
        print("Error: RUNPOD_SSH_PUBLIC_KEY not set. Add it to your .env file.")
        print("Generate a key with: ssh-keygen -t ed25519 -f ~/.ssh/runpod_ed25519")
        return None

    query = """
    mutation DeployPod($input: PodFindAndDeployOnDemandInput!) {
        podFindAndDeployOnDemand(input: $input) {
            id
            name
            desiredStatus
            imageName
        }
    }
    """

    pod_input = {
        "name": name,
        "imageName": image,
        "gpuTypeId": gpu_type,
        "gpuCount": gpu_count,
        "containerDiskInGb": 20,
        "minVcpuCount": 4,
        "minMemoryInGb": 16,
        "ports": "22/tcp,8888/http",
        "volumeMountPath": "/workspace",
        "dockerArgs": build_docker_args(cmd),
        "env": [
            {"key": "PUBLIC_KEY", "value": SSH_PUBLIC_KEY},
            {"key": "HF_HOME", "value": "/workspace/huggingface"},
            {"key": "TRANSFORMERS_CACHE", "value": "/workspace/huggingface"},
        ],
        "supportPublicIp": True,
        "cloudType": "SECURE",
    }

    if network_volume_id:
        pod_input["networkVolumeId"] = network_volume_id
    else:
        pod_input["volumeInGb"] = volume_gb

    variables = {"input": pod_input}

    result = graphql_request(query, variables)
    if "errors" in result:
        print(f"Error creating pod: {result['errors']}")
        return None
    return result["data"]["podFindAndDeployOnDemand"]


def get_pod_status(pod_id: str) -> dict:
    """Get current status of a pod."""
    query = f"""
    query {{
        pod(input: {{podId: "{pod_id}"}}) {{
            id
            name
            desiredStatus
            runtime {{
                uptimeInSeconds
                ports {{
                    ip
                    isIpPublic
                    privatePort
                    publicPort
                    type
                }}
                gpus {{
                    id
                    gpuUtilPercent
                    memoryUtilPercent
                }}
            }}
        }}
    }}
    """
    result = graphql_request(query)
    return result["data"]["pod"]


def wait_for_pod(pod_id: str, timeout: int = 600) -> dict:
    """Wait for pod to be ready with SSH access."""
    print(f"Waiting for pod {pod_id} to be ready...")
    start = time.time()

    while time.time() - start < timeout:
        status = get_pod_status(pod_id)

        if status and status.get("runtime"):
            runtime = status["runtime"]
            if runtime.get("ports"):
                for port in runtime["ports"]:
                    if port["privatePort"] == 22 and port.get("publicPort"):
                        print("\nPod ready!")
                        return status

        print(".", end="", flush=True)
        time.sleep(5)

    print(f"\nTimeout waiting for pod after {timeout}s")
    return None


def get_ssh_command(pod_status: dict, ssh_key: str) -> str:
    """Extract SSH connection command from pod status."""
    if not pod_status or not pod_status.get("runtime"):
        return None

    for port in pod_status["runtime"]["ports"]:
        if port["privatePort"] == 22:
            ip = port["ip"]
            public_port = port["publicPort"]
            return f"ssh -i {ssh_key} root@{ip} -p {public_port}"

    return None


def stop_pod(pod_id: str):
    """Stop a running pod."""
    query = f"""
    mutation {{
        podStop(input: {{podId: "{pod_id}"}}) {{
            id
            desiredStatus
        }}
    }}
    """
    result = graphql_request(query)
    return result


def resume_pod(pod_id: str, gpu_count: int = 1):
    """Resume a stopped (EXITED) pod. Reuses the original GPU type."""
    query = f"""
    mutation {{
        podResume(input: {{podId: "{pod_id}", gpuCount: {gpu_count}}}) {{
            id
            desiredStatus
        }}
    }}
    """
    result = graphql_request(query)
    return result


def terminate_pod(pod_id: str):
    """Terminate (delete) a pod."""
    query = f"""
    mutation {{
        podTerminate(input: {{podId: "{pod_id}"}})
    }}
    """
    result = graphql_request(query)
    return result


def list_pods() -> list:
    """List all current pods."""
    query = """
    query {
        myself {
            pods {
                id
                name
                desiredStatus
                runtime {
                    uptimeInSeconds
                    ports {
                        ip
                        privatePort
                        publicPort
                    }
                }
            }
        }
    }
    """
    result = graphql_request(query)
    return result["data"]["myself"]["pods"]


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="RunPod GPU Instance Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  %(prog)s balance                          Check account balance
  %(prog)s gpus                             List available GPUs and pricing
  %(prog)s volumes                          List network volumes
  %(prog)s launch --image myimg:latest --ssh-key ~/.ssh/id_ed25519
  %(prog)s launch --image myimg:latest --ssh-key ~/.ssh/id_ed25519 --gpu recommended
  %(prog)s launch --image myimg:latest --ssh-key ~/.ssh/id_ed25519 --network-volume vol_abc123
  %(prog)s launch --image myimg:latest --ssh-key ~/.ssh/id_ed25519 --cmd "python train.py"
  %(prog)s list --ssh-key ~/.ssh/id_ed25519
  %(prog)s stop --pod-id abc123
  %(prog)s resume --pod-id abc123 --ssh-key ~/.ssh/id_ed25519
  %(prog)s terminate --pod-id abc123
""")
    parser.add_argument("action", choices=["launch", "status", "list", "stop", "resume", "terminate", "gpus", "balance", "volumes"],
                       help="Action to perform")
    parser.add_argument("--pod-id", help="Pod ID for status/stop/terminate actions")
    parser.add_argument("--image", help="Docker image to use (required for launch)")
    parser.add_argument("--ssh-key", help="Path to SSH private key (required for launch/list)")
    parser.add_argument("--name", default="runpod-dev", help="Pod name (default: runpod-dev)")
    parser.add_argument("--gpu", choices=list(GPU_PRESETS.keys()), default="value",
                       help="GPU preset tier (default: value)")
    parser.add_argument("--gpu-type", help="Exact GPU type ID (overrides --gpu). Run 'gpus' to see options.")
    parser.add_argument("--gpu-count", type=int, default=1, help="Number of GPUs (default: 1)")
    parser.add_argument("--volume", type=int, default=50, help="Pod volume size in GB (default: 50). Ignored if --network-volume is set.")
    parser.add_argument("--network-volume", metavar="VOLUME_ID",
                       help="Attach an existing network volume by ID (overrides --volume). Use 'volumes' action to list available volumes.")
    parser.add_argument("--cmd",
                       help="Command to run in job mode (pod exits when done). Omit for interactive mode (sshd + sleep).")
    parser.add_argument("-y", "--yes", action="store_true",
                       help="Skip cost confirmation prompt (for scripted use)")

    args = parser.parse_args()

    if args.action == "balance":
        balance = check_balance()
        print(f"Current balance: ${balance:.2f}")

    elif args.action == "gpus":
        gpus = list_available_gpus()
        print(f"{'GPU Type':<35} {'VRAM':<10} {'Community':<12} {'Secure':<12}")
        print("-" * 70)
        for gpu in sorted(gpus, key=lambda x: x.get("communityPrice") or 999):
            name = gpu["displayName"] or gpu["id"]
            mem = f"{gpu['memoryInGb']}GB"
            community = f"${gpu['communityPrice']:.2f}/hr" if gpu.get("communityPrice") else "N/A"
            secure = f"${gpu['securePrice']:.2f}/hr" if gpu.get("securePrice") else "N/A"
            print(f"{name:<35} {mem:<10} {community:<12} {secure:<12}")

    elif args.action == "volumes":
        try:
            volumes = list_network_volumes()
        except requests.exceptions.HTTPError as e:
            print(f"Error listing network volumes: {e}")
            print("Note: Your API key may not have permission to list network volumes.")
            sys.exit(1)

        if not volumes:
            print("No network volumes found.")
            print("Create one at: https://www.runpod.io/console/user/storage")
        else:
            print(f"{'ID':<28} {'Name':<20} {'Size':<10} {'Region':<15}")
            print("-" * 75)
            for vol in volumes:
                vol_id = vol.get("id", "?")
                name = vol.get("name", "?")
                size = f"{vol.get('size', '?')}GB"
                region = vol.get("dataCenterId", "?")
                print(f"{vol_id:<28} {name:<20} {size:<10} {region:<15}")

    elif args.action == "list":
        if not args.ssh_key:
            print("Error: --ssh-key is required to display SSH commands.")
            sys.exit(1)
        pods = list_pods()
        if not pods:
            print("No pods found.")
        else:
            for pod in pods:
                ssh_cmd = ""
                if pod.get("runtime") and pod["runtime"].get("ports"):
                    for port in pod["runtime"]["ports"]:
                        if port["privatePort"] == 22:
                            ssh_cmd = f"ssh -i {args.ssh_key} root@{port['ip']} -p {port['publicPort']}"
                print(f"ID: {pod['id']}")
                print(f"  Name: {pod['name']}")
                print(f"  Status: {pod['desiredStatus']}")
                if ssh_cmd:
                    print(f"  SSH: {ssh_cmd}")
                print()

    elif args.action == "launch":
        if not args.image:
            print("Error: --image is required for launch.")
            print("Example: --image runpod/pytorch:2.1.0-py3.10-cuda12.1.0-devel-ubuntu22.04")
            sys.exit(1)
        if not args.ssh_key:
            print("Error: --ssh-key is required for launch.")
            print("Example: --ssh-key ~/.ssh/runpod_ed25519")
            sys.exit(1)

        balance = check_balance()
        print(f"Current balance: ${balance:.2f}")

        gpu_type = args.gpu_type if args.gpu_type else GPU_PRESETS[args.gpu]

        # Cost confirmation
        price = get_gpu_price(gpu_type)
        price_per_gpu = price * args.gpu_count if price else None

        if not args.network_volume:
            print("\nWarning: No --network-volume specified. Using a pod volume instead.")
            print("  Pod volumes are DELETED on terminate. Use a network volume for persistent data.")
            print("  List volumes: python scripts/runpod_launch.py volumes\n")

        storage_desc = f"Network volume {args.network_volume}" if args.network_volume else f"{args.volume}GB pod volume (ephemeral)"
        mode_desc = f"Job: {args.cmd}" if args.cmd else "Interactive (SSH)"

        print(f"\nLaunching pod '{args.name}' with {gpu_type} x{args.gpu_count}")
        if price_per_gpu:
            daily = price_per_gpu * 24
            print(f"  Estimated cost: ~${price_per_gpu:.2f}/hr (${daily:.2f}/day)")
        else:
            print("  Estimated cost: unknown (custom GPU type)")
        print(f"  Image: {args.image}")
        print(f"  Storage: {storage_desc}")
        print(f"  Mode: {mode_desc}")

        if not args.yes:
            try:
                confirm = input("Proceed? [Y/n] ").strip().lower()
                if confirm and confirm != "y":
                    print("Aborted.")
                    sys.exit(0)
            except (EOFError, KeyboardInterrupt):
                print("\nAborted.")
                sys.exit(0)

        pod = create_pod(
            gpu_type, args.image, args.name, args.volume, args.gpu_count,
            network_volume_id=args.network_volume,
            cmd=args.cmd,
        )
        if not pod:
            print("Failed to create pod")
            sys.exit(1)

        print(f"Pod created: {pod['id']}")

        status = wait_for_pod(pod["id"])
        if status:
            ssh_cmd = get_ssh_command(status, args.ssh_key)
            print(f"\n{'='*60}")
            print("Pod ready!")
            print(f"SSH: {ssh_cmd}")
            print(f"{'='*60}")

            # Save pod info
            info = {
                "pod_id": pod["id"],
                "ssh_command": ssh_cmd,
                "gpu": gpu_type,
                "image": args.image,
            }
            if args.network_volume:
                info["network_volume"] = args.network_volume

            # Use config for secure path, with fallback
            try:
                from config import RUNPOD_CONFIG_PATH
                info_path = RUNPOD_CONFIG_PATH
            except ImportError:
                info_path = Path(os.environ.get("CAML_SECURE", Path(__file__).parent.parent / "secure")) / "runpod_current.json"

            info_path.parent.mkdir(exist_ok=True)
            with open(info_path, "w") as f:
                json.dump(info, f, indent=2)
            print(f"Pod info saved to {info_path}")

    elif args.action == "status":
        if not args.pod_id:
            print("--pod-id required for status")
            sys.exit(1)
        status = get_pod_status(args.pod_id)
        print(json.dumps(status, indent=2))

    elif args.action == "stop":
        if not args.pod_id:
            print("--pod-id required for stop")
            sys.exit(1)
        result = stop_pod(args.pod_id)
        print(f"Pod stopped: {result}")

    elif args.action == "resume":
        if not args.pod_id:
            print("--pod-id required for resume")
            sys.exit(1)
        result = resume_pod(args.pod_id, gpu_count=args.gpu_count)
        if result.get("errors"):
            print(f"Resume failed: {result['errors']}")
            sys.exit(1)
        print(f"Pod resume requested: {result['data']['podResume']}")
        # Wait for SSH and print connection details (matches launch behavior)
        if args.ssh_key:
            status = wait_for_pod(args.pod_id)
            if status:
                ssh_cmd = get_ssh_command(status, args.ssh_key)
                print()
                print("=" * 60)
                print("Pod resumed!")
                if ssh_cmd:
                    print(f"SSH: {ssh_cmd}")
                print("=" * 60)

    elif args.action == "terminate":
        if not args.pod_id:
            print("--pod-id required for terminate")
            sys.exit(1)
        result = terminate_pod(args.pod_id)
        print(f"Pod terminated: {result}")


if __name__ == "__main__":
    main()
