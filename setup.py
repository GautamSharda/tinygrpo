import os
import subprocess
import sys

def run_command(command):
    result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

# Clone repositories
run_command("git clone https://github.com/gautamsharda/tinygrpo.git")
run_command("git clone https://github.com/tinygrad/tinygrad.git")

# Set up virtual environment
run_command("python -m venv venv")
run_command("source venv/bin/activate")

# Install tinygrad
os.chdir("tinygrad")
run_command("pip install -e .")
os.chdir("..")

# Install required packages
run_command("pip install safetensors torch transformers numpy")

# Update and install git-lfs
run_command("sudo apt update")
run_command("sudo apt install -y git-lfs")

# Clone model repository
run_command("git clone https://huggingface.co/Qwen/Qwen3-0.6B-Base")

# Run the script
run_command("python qwen3.py")