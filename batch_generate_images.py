import csv
import os
import time
import fal_client
from pathlib import Path

# Configuration
CSV_FILE = "Ten Archaeological Discoveries That Challenge the Timeline of Human History.csv"
OUTPUT_DIR = "generated_images"
PROMPT_COLUMN_INDEX = 2  # Third column (0-indexed)
RESULT_COLUMN_INDEX = 3  # Fourth column for results

# Create output directory if it doesn't exist
Path(OUTPUT_DIR).mkdir(exist_ok=True)

def should_skip_row(row):
    """Check if row should be skipped (starts with 'Scene #')"""
    if not row or len(row) == 0:
        return True
    first_cell = str(row[0]).strip()
    return first_cell.startswith("Scene #") or first_cell == "Scene #"

def generate_image(prompt, output_filename):
    """Generate image using fal.ai Flux Schnell"""
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
            print(f"  Image URL: {image_url}")
            
            # Download the image
            import requests
            response = requests.get(image_url, timeout=30)
            if response.status_code == 200:
                output_path = os.path.join(OUTPUT_DIR, output_filename)
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                print(f"  Downloaded: {len(response.content)} bytes")
                return {"path": output_path, "url": image_url}
            else:
                print(f"  Download failed: HTTP {response.status_code}")
                return {"path": None, "url": image_url}
        
        return None
    except Exception as e:
        print(f"  Error generating image: {e}")
        return None

def main():
    # Read CSV
    rows = []
    with open(CSV_FILE, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)
    
    print(f"Total rows in CSV: {len(rows)}")
    
    # Process first 150 rows (excluding header and Scene # rows)
    processed_count = 0
    
    for i, row in enumerate(rows):
        # Skip header row
        if i == 0:
            continue
            
        # Skip Scene # rows
        if should_skip_row(row):
            print(f"Row {i}: Skipping (Scene # row)")
            continue
        
        # Stop after 150 valid rows
        if processed_count >= 150:
            print(f"\nReached 150 images. Stopping.")
            break
        
        # Get prompt from third column
        if len(row) <= PROMPT_COLUMN_INDEX:
            print(f"Row {i}: Skipping (no prompt in column 3)")
            continue
        
        prompt = row[PROMPT_COLUMN_INDEX].strip()
        if not prompt:
            print(f"Row {i}: Skipping (empty prompt)")
            continue
        
        # Generate filename
        scene_number = row[0].strip() if row[0] else f"row_{i}"
        output_filename = f"scene_{scene_number}.png"
        
        print(f"\nRow {i} (Scene {scene_number}): Generating image...")
        print(f"Prompt: {prompt[:100]}...")
        
        # Generate image
        result = generate_image(prompt, output_filename)
        
        if result:
            # Update result column with both local path AND URL
            while len(row) <= RESULT_COLUMN_INDEX:
                row.append("")
            
            # Store both path and URL separated by " | "
            if result['path']:
                row[RESULT_COLUMN_INDEX] = f"{result['path']} | {result['url']}"
                print(f"✓ Saved: {result['path']}")
                print(f"  URL: {result['url']}")
            else:
                row[RESULT_COLUMN_INDEX] = f"DOWNLOAD_FAILED | {result['url']}"
                print(f"✓ URL saved (download failed): {result['url']}")
            
            processed_count += 1
        else:
            print(f"✗ Failed to generate image")
        
        # Small delay to avoid rate limiting
        time.sleep(1)
    
    # Write updated CSV
    with open(CSV_FILE, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    
    print(f"\n{'='*50}")
    print(f"Batch generation complete!")
    print(f"Total images generated: {processed_count}")
    print(f"Images saved to: {OUTPUT_DIR}/")
    print(f"CSV updated with results")

if __name__ == "__main__":
    main()
