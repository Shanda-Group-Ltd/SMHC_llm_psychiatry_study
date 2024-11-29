import os
import subprocess

def run_classifier_simple(fn):
    current_dir = os.getcwd()
    for root, dirs, files in os.walk(current_dir):
        for dir in dirs:
            subfolder_path = os.path.join(root, dir)
            os.chdir(subfolder_path)
            if fn in os.listdir() and "/classification_code" in subfolder_path:
                print(f"===============Running {fn} in {subfolder_path}===============")
                try:
                    subprocess.run(["python", f"{fn}", "--fusion_type", "late"], check=True) #, "--skip_single_feature_set"], check=True)
                    print(f"Successfully ran {fn} in {subfolder_path}")
                except subprocess.CalledProcessError as e:
                    print(f"Failed to run {fn} in {subfolder_path}: {e}")
            os.chdir(current_dir)

if __name__ == "__main__":
    fn = "classifier_late_fusion.py"
    run_classifier_simple(fn)
