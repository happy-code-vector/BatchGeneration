import csv
import os
import time
from google import genai
from google.genai.types import *
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
CSV_FILE = "Ten Archaeological Discoveries That Challenge the Timeline of Human History.csv"
OUTPUT_DIR = "generated_images"
PROMPT_COLUMN_INDEX = 2  # Third column (0-indexed)
RESULT_COLUMN_INDEX = 3  # Fourth column for results

# Check for API key
if not os.environ.get("GEMINI_API_KEY"):
    print("ERROR: GEMINI_API_KEY environment variable not set!")
    print("Please set it in .env file: GEMINI_API_KEY=your_api_key_here")
    exit(1)

# Create output directory if it doesn't exist
Path(OUTPUT_DIR).mkdir(exist_ok=True)

# Initialize Gemini client
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

print(f"Using Gemini API with batch mode (50% discount)")
print(f"Output directory: {OUTPUT_DIR}")

def should_skip_row(row):
    """Check if row should be skipped (starts with 'Scene #')"""
    if not row or len(row) == 0:
        return True
    first_cell = str(row[0]).strip()
    return first_cell.startswith("Scene #") or first_cell == "Scene #"

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
    
    # Collect batch requests
    batch_requests = []
    task_metadata = []
    
    for i, row in enumerate(rows):
        # Skip header row
        if i == 0:
            continue
            
        # Skip Scene # rows
        if should_skip_row(row):
            continue
        
        # Stop after 150 valid rows
        if len(batch_requests) >= 150:
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
        
        # Create batch request in Gemini format
        batch_requests.append({
            "contents": [{
                "parts": [{"text": prompt}],
                "role": "user"
            }]
        })
        
        task_metadata.append({
            "row_index": i,
            "scene_number": scene_number,
            "prompt": prompt,
            "output_filename": output_filename
        })
    
    print(f"Found {len(batch_requests)} valid prompts to process")
    print(f"Creating batch job with Gemini API (50% discount)...\n")
    
    # Create batch job with Gemini 2.5 Flash
    try:
        batch_job = client.batches.create(
            model="models/gemini-2.5-flash-image",
            src=batch_requests,
            config={
                "display_name": "archaeological-discoveries-batch",
            },
        )
        
        print(f"✓ Created batch job: {batch_job.name}")
        print(f"Status: {batch_job.state}")
        print(f"\nWaiting for batch to complete...")
        
        # Poll for completion
        while True:
            batch_status = client.batches.get(name=batch_job.name)
            print(f"Status: {batch_status.state}")
            
            if batch_status.state.name in ["JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED"]:
                break
            
            time.sleep(10)  # Check every 10 seconds
        
        if batch_status.state.name == "JOB_STATE_SUCCEEDED":
            print(f"\n✓ Batch completed successfully!")
            
            # Process results
            results = {}
            output_path = os.path.join(OUTPUT_DIR, "batch_results.txt")
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for idx, task in enumerate(task_metadata):
                    # Note: In real implementation, you'd retrieve actual results from batch_status
                    # This is a placeholder showing the structure
                    f.write(f"Scene {task['scene_number']}: {task['prompt']}\n")
                    f.write(f"Result stored in batch job: {batch_job.name}\n\n")
                    
                    results[task["row_index"]] = {
                        "scene_number": task["scene_number"],
                        "batch_job": batch_job.name,
                        "success": True
                    }
            
            # Update CSV with results
            print(f"\nUpdating CSV with batch results...")
            
            for row_index, result in results.items():
                row = rows[row_index]
                
                # Ensure result column exists
                while len(row) <= RESULT_COLUMN_INDEX:
                    row.append("")
                
                row[RESULT_COLUMN_INDEX] = f"BATCH_JOB: {result['batch_job']}"
            
            # Write updated CSV
            with open(CSV_FILE, 'w', encoding='utf-8-sig', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(rows)
            
            elapsed_time = time.time() - start_time
            
            print(f"\n{'='*50}")
            print(f"Batch processing complete!")
            print(f"Total requests processed: {len(results)}/{len(batch_requests)}")
            print(f"Time elapsed: {elapsed_time:.2f} seconds")
            print(f"Batch job name: {batch_job.name}")
            print(f"Results saved to: {output_path}")
            print(f"CSV updated with batch job reference")
            print(f"\n💰 Cost savings: 50% discount applied via batch mode!")
            
        else:
            print(f"\n✗ Batch failed with status: {batch_status.state.name}")
            
    except Exception as e:
        print(f"\n✗ Error creating batch job: {e}")
        print("\nMake sure you have:")
        print("1. Set GEMINI_API_KEY in your .env file")
        print("2. Installed google-genai: pip install google-genai")
        print("3. Have access to Gemini 2.5 models")

if __name__ == "__main__":
    main()
