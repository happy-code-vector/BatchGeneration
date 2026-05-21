"""
Scene-based Batch Image Generation
Generates images from CSV files with Scene, Shot_Type, Shot_Title, and Full_Prompt fields
Uses Google Gemini Batch API (50% discount)
"""

import csv
import os
import sys
import time
import datetime
import argparse
from pathlib import Path
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

# Configuration
BATCH_SIZE = 50  # Gemini batch API limit
OUTPUT_DIR_TEMPLATE = "generated_images_{timestamp}_{project}"

# Check for API key
if not os.environ.get("GEMINI_API_KEY"):
    print("ERROR: GEMINI_API_KEY environment variable not set!")
    print("Please set it in .env file: GEMINI_API_KEY=your_api_key_here")
    sys.exit(1)

# Initialize Gemini client
from google import genai
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


def create_output_dir(project_name: str) -> str:
    """Create timestamped output directory"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_project = re.sub(r'[^\w\-]', '_', project_name)[:30]
    output_dir = OUTPUT_DIR_TEMPLATE.format(timestamp=timestamp, project=safe_project)
    Path(output_dir).mkdir(exist_ok=True)
    return output_dir


def sanitize_filename(name: str) -> str:
    """Create safe filename from string"""
    # Remove or replace unsafe characters
    safe = re.sub(r'[<>:"/\\|?*]', '', name)
    # Replace multiple spaces/hyphens with single hyphen
    safe = re.sub(r'[\s_]+', '-', safe)
    safe = re.sub(r'-+', '-', safe)
    # Remove leading/trailing hyphens
    safe = safe.strip('-')
    return safe


def read_csv_prompts(csv_file: str) -> list:
    """Read CSV with Scene, Shot_Type, Shot_Title, Full_Prompt fields"""
    prompts = []

    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        print(f"Total rows in CSV: {len(rows)}")

        for i, row in enumerate(rows, start=1):
            scene = row.get('Scene', '').strip()
            shot_type = row.get('Shot_Type', '').strip()
            shot_title = row.get('Shot_Title', '').strip()
            full_prompt = row.get('Full_Prompt (Copy & Paste Ready)', '').strip()

            if not full_prompt:
                full_prompt = row.get('Full_Prompt', '').strip()

            if scene and shot_type and shot_title and full_prompt:
                # Extract first letter of shot type
                shot_type_letter = shot_type[0].upper() if shot_type else 'U'

                # Create filename: scene - shot_type_letter - shot_title
                safe_title = sanitize_filename(shot_title)
                filename = f"{scene} - {shot_type_letter} - {safe_title}.png"

                prompts.append({
                    'row_index': i,
                    'scene': scene,
                    'shot_type': shot_type,
                    'shot_type_letter': shot_type_letter,
                    'shot_title': shot_title,
                    'full_prompt': full_prompt,
                    'output_filename': filename
                })

    except UnicodeDecodeError:
        print("UTF-8 failed, trying with latin-1 encoding...")
        with open(csv_file, 'r', encoding='latin-1') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        for i, row in enumerate(rows, start=1):
            scene = row.get('Scene', '').strip()
            shot_type = row.get('Shot_Type', '').strip()
            shot_title = row.get('Shot_Title', '').strip()
            full_prompt = row.get('Full_Prompt (Copy & Paste Ready)', '').strip()

            if not full_prompt:
                full_prompt = row.get('Full_Prompt', '').strip()

            if scene and shot_type and shot_title and full_prompt:
                shot_type_letter = shot_type[0].upper() if shot_type else 'U'
                safe_title = sanitize_filename(shot_title)
                filename = f"{scene} - {shot_type_letter} - {safe_title}.png"

                prompts.append({
                    'row_index': i,
                    'scene': scene,
                    'shot_type': shot_type,
                    'shot_type_letter': shot_type_letter,
                    'shot_title': shot_title,
                    'full_prompt': full_prompt,
                    'output_filename': filename
                })

    return prompts


def get_batches(prompts: list, batch_size: int) -> list:
    """Split prompts into batches"""
    batches = []
    for i in range(0, len(prompts), batch_size):
        batches.append(prompts[i:i + batch_size])
    return batches


def list_batches(prompts: list, batch_size: int, csv_file: str):
    """List all available batches"""
    batches = get_batches(prompts, batch_size)
    total_batches = len(batches)
    total_prompts = len(prompts)

    print(f"\n{'='*60}")
    print(f"CSV File: {csv_file}")
    print(f"Total prompts: {total_prompts}")
    print(f"Batch size: {batch_size}")
    print(f"Total batches: {total_batches}")
    print(f"{'='*60}\n")

    for i, batch in enumerate(batches, start=1):
        start_idx = batch[0]['row_index']
        end_idx = batch[-1]['row_index']
        preview = ", ".join([f"{p['scene']}-{p['shot_type_letter']}" for p in batch[:5]])
        if len(batch) > 5:
            preview += "..."
        print(f"Batch {i}: Rows {start_idx}-{end_idx} ({len(batch)} images)")
        print(f"       {preview}")
        print()

    return total_batches


def process_batch(batch: list, output_dir: str, batch_num: int, total_batches: int):
    """Process a single batch using Gemini batch API"""
    print(f"\n{'='*60}")
    print(f"Processing Batch {batch_num}/{total_batches} ({len(batch)} images)")
    print(f"{'='*60}\n")

    # Prepare batch requests
    batch_requests = []
    task_metadata = []

    for item in batch:
        # Use full prompt directly
        prompt_text = item['full_prompt']

        batch_requests.append({
            "contents": [{
                "parts": [{"text": prompt_text}],
                "role": "user"
            }]
        })

        task_metadata.append({
            "row_index": item['row_index'],
            "scene": item['scene'],
            "shot_type": item['shot_type'],
            "shot_type_letter": item['shot_type_letter'],
            "shot_title": item['shot_title'],
            "full_prompt": prompt_text,
            "output_filename": item['output_filename']
        })

    # Create batch job with Gemini 2.5 Flash
    print(f"Creating batch job with Gemini API (50% discount)...")
    try:
        batch_job = client.batches.create(
            model="models/gemini-2.5-flash-image",
            src=batch_requests,
            config={
                "display_name": f"scene-batch-{batch_num}",
            },
        )

        print(f"✓ Created batch job: {batch_job.name}")
        print(f"Status: {batch_job.state}")
        print(f"\nWaiting for batch to complete...")

        # Poll for completion
        count = 0
        while True:
            batch_status = client.batches.get(name=batch_job.name)
            print(f"Status: {batch_status.state.name} ({count})")
            count += 1

            if batch_status.state.name in ["JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED"]:
                break

            time.sleep(10)  # Check every 10 seconds

        if batch_status.state.name == "JOB_STATE_SUCCEEDED":
            print(f"\n✓ Batch completed successfully!")

            # Process results
            success_count = 0
            error_count = 0

            if batch_status.dest and batch_status.dest.inlined_responses:
                print("Processing inline results...")

                for i, inline_response in enumerate(batch_status.dest.inlined_responses):
                    task = task_metadata[i]
                    scene = task['scene']
                    shot_letter = task['shot_type_letter']
                    title = task['shot_title'][:30]

                    print(f"  Processing {scene}-{shot_letter}: {title}...")

                    # Check for a successful response with image data
                    if inline_response.response:
                        try:
                            image_parts = [
                                part.inline_data.data
                                for part in inline_response.response.parts
                                if part.inline_data
                            ]

                            if image_parts:
                                # Save the image
                                image_path = os.path.join(output_dir, task['output_filename'])

                                # Get the first part with image data
                                for part in inline_response.response.parts:
                                    if part.inline_data:
                                        image = part.as_image()
                                        image.save(image_path)
                                        break

                                # Save prompt info
                                prompt_file = os.path.join(output_dir, f"{task['output_filename']}.prompt.txt")
                                with open(prompt_file, 'w', encoding='utf-8') as f:
                                    f.write(f"Scene: {task['scene']}\n")
                                    f.write(f"Shot Type: {task['shot_type']}\n")
                                    f.write(f"Shot Title: {task['shot_title']}\n")
                                    f.write(f"Filename: {task['output_filename']}\n")
                                    f.write(f"\nPrompt:\n{task['full_prompt']}\n")

                                print(f"    ✓ Saved: {task['output_filename']}")
                                success_count += 1
                            else:
                                print(f"    ✗ No image data in response")
                                error_count += 1

                        except Exception as e:
                            print(f"    ✗ Error: {e}")
                            error_count += 1
                    elif inline_response.error:
                        print(f"    ✗ API Error: {inline_response.error}")
                        error_count += 1
                    else:
                        print(f"    ✗ No response received")
                        error_count += 1

            elif batch_status.dest and batch_status.dest.file_name:
                print(f"Results are in file: {batch_status.dest.file_name}")
                print("Downloading result file content...")
                file_content = client.files.download(file=batch_status.dest.file_name)

                output_path = os.path.join(output_dir, f"batch_{batch_num}_results.json")
                with open(output_path, 'wb') as f:
                    f.write(file_content)

                print(f"Results downloaded to: {output_path}")
                success_count = len(batch_requests)
            else:
                print("No results found (neither file nor inline).")
                error_count = len(batch_requests)

            return {
                'batch_num': batch_num,
                'success': success_count,
                'errors': error_count,
                'total': len(batch_requests)
            }

        else:
            print(f"\n✗ Batch failed with status: {batch_status.state.name}")
            return {
                'batch_num': batch_num,
                'success': 0,
                'errors': len(batch_requests),
                'total': len(batch_requests)
            }

    except Exception as e:
        print(f"\n✗ Error creating batch job: {e}")
        return {
            'batch_num': batch_num,
            'success': 0,
            'errors': len(batch_requests),
            'total': len(batch_requests),
            'error': str(e)
        }


def save_results(output_dir: str, all_results: list, total_prompts: int, csv_file: str):
    """Save generation results to a file"""
    results_file = os.path.join(output_dir, "generation_results.txt")

    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("SCENE BATCH GENERATION RESULTS\n")
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"CSV File: {csv_file}\n")
        f.write(f"Total prompts: {total_prompts}\n")
        f.write(f"{'='*60}\n\n")

        total_success = 0
        total_errors = 0

        for result in all_results:
            batch_num = result['batch_num']
            success = result['success']
            errors = result['errors']
            total = result['total']

            f.write(f"{'='*60}\n")
            f.write(f"Batch {batch_num}\n")
            f.write(f"{'='*60}\n")
            f.write(f"Success: {success}/{total}\n")
            f.write(f"Errors: {errors}/{total}\n")

            if 'error' in result:
                f.write(f"Error: {result['error']}\n")

            total_success += success
            total_errors += errors

        f.write(f"\n\n{'='*60}\n")
        f.write("FINAL SUMMARY\n")
        f.write(f"{'='*60}\n")
        f.write(f"Total prompts: {total_prompts}\n")
        f.write(f"Total success: {total_success}\n")
        f.write(f"Total errors: {total_errors}\n")
        f.write(f"Success rate: {total_success/total_prompts*100:.1f}%\n")
        f.write(f"\nCost savings: 50% discount applied via Gemini batch mode!\n")

    print(f"\n✓ Results saved to: {results_file}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Generate scene images using Gemini Batch API'
    )
    parser.add_argument(
        '--csv',
        type=str,
        required=True,
        help='CSV file to read (Scene, Shot_Type, Shot_Title, Full_Prompt format)'
    )
    parser.add_argument(
        '--batch',
        type=str,
        help='Batch number to generate (e.g., "1") or "all" for all batches'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available batches without generating'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=BATCH_SIZE,
        help=f'Number of images per batch (default: {BATCH_SIZE}, max: 50 for Gemini)'
    )
    parser.add_argument(
        '--count',
        type=int,
        help='Number of images to generate (e.g., 10, 50, 100)'
    )

    args = parser.parse_args()

    # Validate batch size
    if args.batch_size > 50:
        print("WARNING: Gemini batch API has a limit of 50 requests per batch.")
        print(f"Setting batch size to 50.")
        args.batch_size = 50

    # Check if CSV file exists
    if not os.path.exists(args.csv):
        print(f"ERROR: CSV file not found: {args.csv}")
        sys.exit(1)

    # Get project name from CSV filename
    project_name = Path(args.csv).stem

    # Read prompts from CSV
    print(f"Reading prompts from: {args.csv}")
    prompts = read_csv_prompts(args.csv)

    if not prompts:
        print("ERROR: No prompts found in CSV file!")
        sys.exit(1)

    print(f"Found {len(prompts)} scenes")

    # Handle --count option
    if args.count:
        if args.count > len(prompts):
            print(f"WARNING: Requested {args.count} images but only {len(prompts)} available.")
            print(f"Generating all {len(prompts)} images.")
            count = len(prompts)
        else:
            count = args.count

        prompts = prompts[:count]
        print(f"Generating first {len(prompts)} images")

    # List batches if requested
    if args.list:
        list_batches(prompts, args.batch_size, args.csv)
        return

    # Get batches
    batches = get_batches(prompts, args.batch_size)
    total_batches = len(batches)

    # Determine which batches to process
    if args.count:
        batches_to_process = list(range(total_batches))
    elif args.batch:
        if args.batch.lower() == 'all':
            batches_to_process = list(range(total_batches))
        else:
            try:
                batch_num = int(args.batch)
                if batch_num < 1 or batch_num > total_batches:
                    print(f"ERROR: Batch number must be between 1 and {total_batches}")
                    list_batches(prompts, args.batch_size, args.csv)
                    sys.exit(1)
                batches_to_process = [batch_num - 1]  # Convert to 0-indexed
            except ValueError:
                print(f"ERROR: Invalid batch number '{args.batch}'. Use a number or 'all'")
                sys.exit(1)
    else:
        # No batch/count specified, ask user
        total_batches = list_batches(prompts, args.batch_size, args.csv)
        user_input = input("\nEnter batch number (1-{}), 'all', or a count: ".format(total_batches)).strip()

        if user_input.lower() == 'all':
            batches_to_process = list(range(total_batches))
        else:
            try:
                # Try as batch number first
                batch_num = int(user_input)
                if batch_num < 1 or batch_num > total_batches:
                    print(f"ERROR: Batch number must be between 1 and {total_batches}")
                    sys.exit(1)
                batches_to_process = [batch_num - 1]
            except ValueError:
                # Try as count
                try:
                    count = int(user_input)
                    if count < 1 or count > len(prompts):
                        print(f"ERROR: Count must be between 1 and {len(prompts)}")
                        sys.exit(1)
                    prompts = prompts[:count]
                    batches = get_batches(prompts, args.batch_size)
                    total_batches = len(batches)
                    batches_to_process = list(range(total_batches))
                    print(f"Generating first {len(prompts)} images")
                except ValueError:
                    print(f"ERROR: Invalid input '{user_input}'")
                    sys.exit(1)

    # Create output directory
    output_dir = create_output_dir(project_name)
    print(f"\nOutput directory: {output_dir}")

    # Process batches
    start_time = time.time()
    all_results = []

    for batch_idx in batches_to_process:
        batch = batches[batch_idx]
        batch_num = batch_idx + 1

        result = process_batch(batch, output_dir, batch_num, total_batches)
        all_results.append(result)

    # Save results
    save_results(output_dir, all_results, len(prompts), args.csv)

    elapsed_time = time.time() - start_time
    total_success = sum(r['success'] for r in all_results)
    total_errors = sum(r['errors'] for r in all_results)

    print(f"\n{'='*60}")
    print(f"BATCH GENERATION COMPLETE!")
    print(f"{'='*60}")
    print(f"Total images: {len(prompts)}")
    print(f"Generated: {total_success}")
    print(f"Errors: {total_errors}")
    print(f"Time elapsed: {elapsed_time:.1f} seconds")
    print(f"Output directory: {output_dir}")
    print(f"\nCost savings: 50% discount applied via Gemini batch mode!")


if __name__ == "__main__":
    main()
