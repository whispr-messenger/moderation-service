import subprocess
import tensorflow as tf

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"

def check_nvidia_driver_and_cuda():
    print("Checking NVIDIA driver & CUDA (nvidia-smi)...")

    code, out, err = _run_cmd(["nvidia-smi"])

    if code != 0:
        print(f"{YELLOW}⚠️ nvidia-smi not available. Is NVIDIA driver installed?{RESET}")
        if err:
            print(f"{YELLOW}{err}{RESET}")
        return None

    print(out)
    driver_version = None
    cuda_version = None

    for line in out.splitlines():
        if "Driver Version" in line and "CUDA Version" in line:
            # e.g.: | NVIDIA-SMI 550.54.14    Driver Version: 550.54.14    CUDA Version: 12.4 |
            parts = line.split("Driver Version:")
            if len(parts) > 1:
                driver_version = parts[1].split()[0]
            parts = line.split("CUDA Version:")
            if len(parts) > 1:
                cuda_version = parts[1].split()[0]
            break

    if driver_version:
        print(f"{GREEN}✅ NVIDIA Driver Version: {driver_version}{RESET}")
    else:
        print(f"{YELLOW}⚠️ Could not parse NVIDIA driver version from nvidia-smi output{RESET}")

    if cuda_version:
        print(f"{GREEN}✅ CUDA Runtime Version (from nvidia-smi): {cuda_version}{RESET}")
    else:
        print(f"{YELLOW}⚠️ Could not parse CUDA runtime version from nvidia-smi output{RESET}")

    return {
        "driver_version": driver_version,
        "cuda_runtime_version": cuda_version,
    }

def check_nvcc():
    print("Checking nvcc (CUDA Toolkit)...")

    code, out, err = _run_cmd(["nvcc", "--version"])

    if code != 0:
        print(f"{YELLOW}⚠️ nvcc not found. CUDA Toolkit may not be installed or not in PATH.{RESET}")
        if err:
            print(f"{YELLOW}{err}{RESET}")
        return None

    print(out)

    cuda_toolkit_version = None
    for line in out.splitlines():
        # e.g.: Cuda compilation tools, release 12.4, V12.4.99
        if "release" in line and "Cuda compilation tools" in line:
            parts = line.split("release")
            if len(parts) > 1:
                cuda_toolkit_version = parts[1].split(",")[0].strip()
            break

    if cuda_toolkit_version:
        print(f"{GREEN}✅ CUDA Toolkit Version (from nvcc): {cuda_toolkit_version}{RESET}")
    else:
        print(f"{YELLOW}⚠️ Could not parse CUDA Toolkit version from nvcc output{RESET}")

    return {
        "cuda_toolkit_version": cuda_toolkit_version,
    }

def check_tf_cuda_build_info():
    print("Checking TensorFlow CUDA build info...")
    try:
        build_info = tf.sysconfig.get_build_info()
        cuda_version = build_info.get("cuda_version", "unknown")
        cudnn_version = build_info.get("cudnn_version", "unknown")
        print(f"{GREEN}✅ TensorFlow built with CUDA: {cuda_version}, cuDNN: {cudnn_version}{RESET}")
        return {
            "tf_cuda_version": cuda_version,
            "tf_cudnn_version": cudnn_version,
        }
    except Exception as e:
        print(f"{YELLOW}⚠️ Could not get TensorFlow CUDA build info: {e}{RESET}")
        return None

def _run_cmd(cmd: list[str]) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except FileNotFoundError:
        return 127, "", f"Command not found: {cmd[0]}"

def check_tf_gpu_usable():
    from tensorflow.python.client import device_lib

    print("Checking TensorFlow GPU usability...")
    devices = device_lib.list_local_devices()
    has_gpu_device = any(d.device_type == "GPU" for d in devices)

    if not has_gpu_device:
        print(f"{YELLOW}⚠️ TensorFlow did not register any GPU device. Using CPU.{RESET}")
        return "cpu"

    try:
        x = tf.random.uniform((1024, 1024))
        y = tf.matmul(x, x)
        _ = y.numpy()
        print(f"{GREEN}✅ TensorFlow GPU computation test passed{RESET}")
        return "gpu"
    except Exception as e:
        print(f"{RED}❌ TensorFlow GPU computation failed: {e}{RESET}")
        print(f"{YELLOW}⚠️ Will fall back to CPU.{RESET}")
        return "cpu"