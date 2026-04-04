import os
import subprocess
import csv
import re
from datetime import datetime

# =====================================================================
# CONFIGURATION
# =====================================================================
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_TOKEN = os.getenv("HF_TOKEN", "") 

# Curated list of ungated, high-reasoning open weights
MODELS_TO_TEST = [
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    "microsoft/Phi-3-mini-4k-instruct",
    "microsoft/Phi-3-small-8k-instruct",
    "NousResearch/Hermes-2-Pro-Llama-3-8B",
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
]

TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
RESULTS_FILE = f"leaderboard_{TIMESTAMP}.csv"
LOG_DIR = f"logs_{TIMESTAMP}"
TIMEOUT_SECONDS = 600

def extract_scores(output_text):
    scores = {"Task 1": 0.0, "Task 2": 0.0, "Task 3": 0.0, "Average": 0.0}
    
    t1_match = re.search(r"Task 1.*:\s*([0-9.]+)\s*/", output_text)
    t2_match = re.search(r"Task 2.*:\s*([0-9.]+)\s*/", output_text)
    t3_match = re.search(r"Task 3.*:\s*([0-9.]+)\s*/", output_text)
    avg_match = re.search(r"Average.*:\s*([0-9.]+)\s*/", output_text)
    
    if t1_match: scores["Task 1"] = float(t1_match.group(1))
    if t2_match: scores["Task 2"] = float(t2_match.group(1))
    if t3_match: scores["Task 3"] = float(t3_match.group(1))
    if avg_match: scores["Average"] = float(avg_match.group(1))
        
    return scores

def run_benchmark():
    if not API_TOKEN:
        print("[WARNING] No API Token found in environment variables.")
        print("Please run: export HF_TOKEN=\"your_token\" before starting.")
        return

    os.makedirs(LOG_DIR, exist_ok=True)

    print(f"Starting Benchmark Run for {len(MODELS_TO_TEST)} models...")
    print(f"Results will be saved to: {RESULTS_FILE}")
    print(f"Detailed logs will be saved in: ./{LOG_DIR}/\n")

    with open(RESULTS_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Model", "Status", "Task 1", "Task 2", "Task 3", "Average Score"])
        
        for idx, model in enumerate(MODELS_TO_TEST, 1):
            print(f"[{idx}/{len(MODELS_TO_TEST)}] Testing: {model} ... ", end="", flush=True)
            
            env = os.environ.copy()
            env["MODEL_NAME"] = model
            env["API_BASE_URL"] = API_BASE_URL
            env["HF_TOKEN"] = API_TOKEN
            env["ENV_URL"] = "http://localhost:7860"
            
            safe_model_name = model.replace("/", "_")
            log_filepath = os.path.join(LOG_DIR, f"{safe_model_name}.txt")
            
            try:
                result = subprocess.run(
                    ["python", "inference.py"],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=TIMEOUT_SECONDS
                )
                
                with open(log_filepath, "w") as log_file:
                    log_file.write(result.stdout)
                    if result.stderr:
                        log_file.write("\n\n--- STDERR ---\n")
                        log_file.write(result.stderr)
                
                if result.returncode == 0:
                    scores = extract_scores(result.stdout)
                    print(f"[SUCCESS] (Avg: {scores['Average']})")
                    writer.writerow([
                        model, "Completed", 
                        scores["Task 1"], scores["Task 2"], scores["Task 3"], scores["Average"]
                    ])
                else:
                    print(f"[FAILED] (Crash/Error) -> Check {log_filepath}")
                    writer.writerow([model, "Error", 0, 0, 0, 0])
                    
            except subprocess.TimeoutExpired:
                print(f"[TIMEOUT] (Exceeded 10 mins) -> Check {log_filepath}")
                writer.writerow([model, "Timeout", 0, 0, 0, 0])
                with open(log_filepath, "w") as log_file:
                    log_file.write("PROCESS TIMED OUT AFTER 10 MINUTES.\n")
            except Exception as e:
                print(f"[ERROR] Unexpected Error: {e}")
                writer.writerow([model, "Crash", 0, 0, 0, 0])

    print(f"\nBenchmark complete. Check {RESULTS_FILE} for your leaderboard.")

if __name__ == "__main__":
    run_benchmark()