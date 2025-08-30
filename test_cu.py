import subprocess
import os

def run_commands():
    commands = [
        "cd intersect_extension",
        "python setup.py install", 
        "cd ..",
        "python train_with_height.py"
    ]
    
    # Execute commands in sequence
    try:
        # Change to intersect_extension directory
        os.chdir("intersect_extension")
        subprocess.run(["python", "setup.py", "install"], check=True)
        
        # Change back to parent directory
        os.chdir("..")
        subprocess.run(["python", "train_with_height.py"], check=True)
        
        print("All commands executed successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error: {e}")
    except FileNotFoundError as e:
        print(f"File or directory not found: {e}")

if __name__ == "__main__":
    run_commands()