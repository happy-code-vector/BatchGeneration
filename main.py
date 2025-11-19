import csv
import os
import time
import datetime
from google import genai
from google.genai.types import *
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image
import io

# Load environment variables from .env file
load_dotenv()

# Configuration
CSV_FILE = "Ten Archaeological Discoveries That Challenge the Timeline of Human History.csv"
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"generated_images_{timestamp}"
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
            
            # Check if results are inline or in a file
            if batch_status.dest and batch_status.dest.inlined_responses:
                print("Processing inline results...")
                
                success_count = 0
                error_count = 0
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(f"GEMINI BATCH API RESULTS\n")
                    f.write(f"Batch Job: {batch_job.name}\n")
                    f.write(f"Total Requests: {len(batch_requests)}\n")
                    f.write(f"{'='*80}\n\n")
                    
                    # Extract responses from the batch job
                    for i, inline_response in enumerate(batch_status.dest.inlined_responses):
                        task = task_metadata[i]
                        scene_num = task['scene_number']
                        
                        f.write(f"\n{'='*80}\n")
                        f.write(f"SCENE {scene_num}\n")
                        f.write(f"{'='*80}\n\n")
                        f.write(f"PROMPT:\n{task['prompt']}\n\n")
                        f.write(f"{'-'*80}\n")
                        f.write(f"RESPONSE:\n{'-'*80}\n\n")
                        
                        # Check for a successful response with image data
                        if inline_response.response:
                            try:
                                # Extract image parts from response
                                image_parts = [
                                    part.inline_data.data
                                    for part in inline_response.response.parts
                                    if part.inline_data
                                ]
                                
                                if image_parts:
                                    # Save the image
                                    image_path = os.path.join(OUTPUT_DIR, task['output_filename'])
                                    
                                    # Get the first part with image data
                                    for part in inline_response.response.parts:
                                        if part.inline_data:
                                            image = part.as_image()
                                            image.save(image_path)
                                            break
                                    
                                    f.write(f"Image saved: {image_path}\n")
                                    f.write(f"[STATUS: SUCCESS]\n")
                                    success_count += 1
                                    print(f"  ✓ Scene {scene_num}: {image_path}")
                                else:
                                    # No image data, might be text response
                                    try:
                                        response_text = inline_response.response.text
                                        f.write(f"Text response: {response_text}\n")
                                        f.write(f"[STATUS: SUCCESS - text only]\n")
                                        success_count += 1
                                    except:
                                        f.write(f"No image data found\n")
                                        f.write(f"[STATUS: ERROR]\n")
                                        error_count += 1
                                        
                            except Exception as e:
                                f.write(f"ERROR processing response: {e}\n")
                                f.write(f"[STATUS: ERROR]\n")
                                error_count += 1
                                print(f"  ✗ Scene {scene_num}: {e}")
                        elif inline_response.error:
                            f.write(f"ERROR: {inline_response.error}\n")
                            f.write(f"[STATUS: ERROR]\n")
                            error_count += 1
                            print(f"  ✗ Scene {scene_num}: {inline_response.error}")
                        else:
                            f.write(f"ERROR: No response received\n")
                            f.write(f"[STATUS: ERROR]\n")
                            error_count += 1
                            print(f"  ✗ Scene {scene_num}: No response")
                    
                    # Summary at the end
                    f.write(f"\n\n{'='*80}\n")
                    f.write(f"SUMMARY\n")
                    f.write(f"{'='*80}\n")
                    f.write(f"Total: {len(batch_requests)}\n")
                    f.write(f"Success: {success_count}\n")
                    f.write(f"Errors: {error_count}\n")
                
                print(f"\n✓ Summary saved to: {output_path}")
                print(f"✓ Images saved to: {OUTPUT_DIR}/")
                print(f"Success: {success_count} | Errors: {error_count}")
            
            elif batch_status.dest and batch_status.dest.file_name:
                print(f"Results are in file: {batch_status.dest.file_name}")
                print("Downloading result file content...")
                file_content = client.files.download(file=batch_status.dest.file_name)
                
                with open(output_path, 'wb') as f:
                    f.write(file_content)
                
                print(f"Results downloaded to: {output_path}")
                # For file-based results, mark all as successful
                for task in task_metadata:
                    results[task["row_index"]] = {
                        "scene_number": task["scene_number"],
                        "response": f"See file: {output_path}",
                        "success": True
                    }
            else:
                print("No results found (neither file nor inline).")
                return
            
            elapsed_time = time.time() - start_time
            
            print(f"\n{'='*50}")
            print(f"Batch image generation complete!")
            print(f"Time elapsed: {elapsed_time:.2f} seconds")
            print(f"Batch job name: {batch_job.name}")
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
