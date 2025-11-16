import csv
import os
import time
import fal_client
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Load environment variables from .env file
load_dotenv()

# Configuration
CSV_FILE = "Ten Archaeological Discoveries That Challenge the Timeline of Human History.csv"
OUTPUT_DIR = "generated_images"
PROMPT_COLUMN_INDEX = 2  # Third column (0-indexed)
RESULT_COLUMN_INDEX = 3  # Fourth column for results
MAX_WORKERS = 10  # Number of concurrent API calls

# Thread-safe lock for printing
print_lock = threading.Lock()

# Check for API key
if not os.environ.get("FAL_KEY"):
    print("ERROR: FAL_KEY environment variable not set!")
    print("Please set it in .env file: FAL_KEY=your_api_key_here")
    exit(1)

# Create output directory if it doesn't exist
Path(OUTPUT_DIR).mkdir(exist_ok=True)

print(f"Using FAL_KEY: {os.environ.get('FAL_KEY')[:10]}...")
print(f"Output directory: {OUTPUT_DIR}")

def should_skip_row(row):
    """Check if row should be skipped (starts with 'Scene #')"""
    if not row or len(row) == 0:
        return True
    first_cell = str(row[0]).strip()
    return first_cell.startswith("Scene #") or first_cell == "Scene #"

def safe_print(*args, **kwargs):
    """Thread-safe print function"""
    with print_lock:
        print(*args, **kwargs)

def generate_image_task(row_index, scene_number, prompt, output_filename):
    """Generate image task for threading"""
    import requests
    
    safe_print(f"\n[Scene {scene_number}] Starting generation...")
    safe_print(f"[Scene {scene_number}] Prompt: {prompt[:80]}...")
    
    try:
        result = fal_client.subscribe(
            "fal-ai/flux/schnell",
            arguments={
                "prompt": prompt,
                "image_size": "landscape_16_9",
                "num_inference_steps": 4,
                "num_images": 1,
                "enable_safety_checker": False
            },
        )
        
        # Get the image URL from result
        if result and 'images' in result and len(result['images']) > 0:
            image_url = result['images'][0]['url']
            safe_print(f"[Scene {scene_number}] Generated! URL: {image_url[:50]}...")
            
            # Download the image
            response = requests.get(image_url, timeout=30)
            if response.status_code == 200:
                output_path = os.path.join(OUTPUT_DIR, output_filename)
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                safe_print(f"[Scene {scene_number}] ✓ Downloaded: {len(response.content)} bytes")
                return {
                    "row_index": row_index,
                    "scene_number": scene_number,
                    "path": output_path,
                    "url": image_url,
                    "success": True
                }
            else:
                safe_print(f"[Scene {scene_number}] ✗ Download failed: HTTP {response.status_code}")
                return {
                    "row_index": row_index,
                    "scene_number": scene_number,
                    "path": None,
                    "url": image_url,
                    "success": False
                }
        
        safe_print(f"[Scene {scene_number}] ✗ No image in result")
        return None
    except Exception as e:
        safe_print(f"[Scene {scene_number}] ✗ Error: {e}")
        return None

def main():
    start_time = time.time()
    
    # Read CSV with proper encoding handling
    rows = []
    try:
        with open(CSV_FILE, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
    except UnicodeDecodeError:
        print("UTF-8 failed, trying with latin-1 encoding...")
        with open(CSV_FILE, 'r', encoding='latin-1') as f:
            reader = csv.reader(f)
            rows = list(reader)
    
    print(f"Total rows in CSV: {len(rows)}")
    print(f"Using {MAX_WORKERS} concurrent workers\n")
    
    # Collect tasks to process
    tasks = []
    for i, row in enumerate(rows):
        # Skip header row
        if i == 0:
            continue
            
        # Skip Scene # rows
        if should_skip_row(row):
            continue
        
        # Stop after 150 valid rows
        if len(tasks) >= 150:
            break
        
        # Get prompt from third column
        if len(row) <= PROMPT_COLUMN_INDEX:
            continue
        
        prompt = row[PROMPT_COLUMN_INDEX].strip()
        if not prompt:
            continue
        
        # Generate filename
        scene_number = row[0].strip() if row[0] else f"row_{i}"
        output_filename = f"scene_{scene_number}.png"
        
        tasks.append({
            "row_index": i,
            "scene_number": scene_number,
            "prompt": prompt,
            "output_filename": output_filename
        })
    
    print(f"Found {len(tasks)} valid prompts to process\n")
    
    # Process tasks with thread pool
    results = {}
    completed = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(
                generate_image_task,
                task["row_index"],
                task["scene_number"],
                task["prompt"],
                task["output_filename"]
            ): task for task in tasks
        }
        
        # Process completed tasks
        for future in as_completed(future_to_task):
            completed += 1
            result = future.result()
            
            if result:
                results[result["row_index"]] = result
                safe_print(f"\n[Progress: {completed}/{len(tasks)}] Scene {result['scene_number']} complete")
            else:
                safe_print(f"\n[Progress: {completed}/{len(tasks)}] Task failed")
    
    # Update CSV with results
    print(f"\n{'='*50}")
    print("Updating CSV with results...")
    
    for row_index, result in results.items():
        row = rows[row_index]
        
        # Ensure result column exists
        while len(row) <= RESULT_COLUMN_INDEX:
            row.append("")
        
        # Store both path and URL
        if result['path']:
            row[RESULT_COLUMN_INDEX] = f"{result['path']} | {result['url']}"
        else:
            row[RESULT_COLUMN_INDEX] = f"DOWNLOAD_FAILED | {result['url']}"
    
    # Write updated CSV
    with open(CSV_FILE, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    
    elapsed_time = time.time() - start_time
    
    print(f"\n{'='*50}")
    print(f"Batch generation complete!")
    print(f"Total images generated: {len(results)}/{len(tasks)}")
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    print(f"Average time per image: {elapsed_time/len(results):.2f} seconds")
    print(f"Images saved to: {OUTPUT_DIR}/")
    print(f"CSV updated with results")

if __name__ == "__main__":
    main()
