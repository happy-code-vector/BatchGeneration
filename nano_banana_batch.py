"""
Nano Banana Batch Image Generation
Generates images in batches from CSV file containing style and prompt pairs
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
from PIL import Image
import io
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Load environment variables
load_dotenv()

# Configuration
CSV_FILE = "simpler.csv"
BATCH_SIZE = 50  # Gemini batch API limit
OUTPUT_DIR_TEMPLATE = "generated_images_{timestamp}"

# Check for API key
if not os.environ.get("GEMINI_API_KEY"):
    print("ERROR: GEMINI_API_KEY environment variable not set!")
    print("Please set it in .env file: GEMINI_API_KEY=your_api_key_here")
    sys.exit(1)

# Initialize Gemini client
from google import genai
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


def create_output_dir() -> str:
    """Create timestamped output directory"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_DIR_TEMPLATE.format(timestamp=timestamp)
    Path(output_dir).mkdir(exist_ok=True)
    return output_dir


def read_csv_prompts(csv_file: str) -> list:
    """Read CSV and extract style and prompt pairs"""
    prompts = []

    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)

        print(f"Total rows in CSV: {len(rows)}")

        # Skip header row (row 0)
        for i, row in enumerate(rows[1:], start=2):  # Start from row 2 (after header)
            if len(row) >= 2:
                style = row[0].strip()
                full_prompt = row[1].strip()

                if style and full_prompt:
                    # Create safe filename from style
                    safe_style = "".join(c for c in style if c.isalnum() or c in (' ', '-', '_')).strip()
                    safe_style = safe_style[:50]  # Limit length

                    prompts.append({
                        'row_index': i,
                        'style': style,
                        'prompt': full_prompt,
                        'safe_style': safe_style
                    })

    except UnicodeDecodeError:
        print("UTF-8 failed, trying with latin-1 encoding...")
        with open(csv_file, 'r', encoding='latin-1') as f:
            reader = csv.reader(f)
            rows = list(reader)

        for i, row in enumerate(rows[1:], start=2):
            if len(row) >= 2:
                style = row[0].strip()
                full_prompt = row[1].strip()

                if style and full_prompt:
                    safe_style = "".join(c for c in style if c.isalnum() or c in (' ', '-', '_')).strip()
                    safe_style = safe_style[:50]

                    prompts.append({
                        'row_index': i,
                        'style': style,
                        'prompt': full_prompt,
                        'safe_style': safe_style
                    })

    return prompts


def get_batches(prompts: list, batch_size: int) -> list:
    """Split prompts into batches"""
    batches = []
    for i in range(0, len(prompts), batch_size):
        batches.append(prompts[i:i + batch_size])
    return batches


def list_batches(prompts: list, batch_size: int):
    """List all available batches"""
    batches = get_batches(prompts, batch_size)
    total_batches = len(batches)
    total_prompts = len(prompts)

    print(f"\n{'='*60}")
    print(f"CSV File: {CSV_FILE}")
    print(f"Total prompts: {total_prompts}")
    print(f"Batch size: {batch_size}")
    print(f"Total batches: {total_batches}")
    print(f"{'='*60}\n")

    for i, batch in enumerate(batches, start=1):
        start_idx = batch[0]['row_index']
        end_idx = batch[-1]['row_index']
        styles_preview = ", ".join([b['safe_style'][:20] for b in batch[:3]])
        if len(batch) > 3:
            styles_preview += "..."
        print(f"Batch {i}: Rows {start_idx}-{end_idx} ({len(batch)} images)")
        print(f"       {styles_preview}")
        print()

    return total_batches


def worker_process_batch(args):
    """Worker function for parallel processing - must be pickle-able"""
    batch, output_dir, batch_num, total_batches, api_key = args

    # Initialize client in this process
    from google import genai
    worker_client = genai.Client(api_key=api_key)

    print(f"\n[Worker {batch_num}] Starting Batch {batch_num}/{total_batches} ({len(batch)} images)")

    # Prepare batch requests
    batch_requests = []
    task_metadata = []

    for item in batch:
        combined_prompt = f"Style: {item['style']}. {item['prompt']}"
        filename = f"row_{item['row_index']}_{item['safe_style']}.png"

        batch_requests.append({
            "contents": [{
                "parts": [{"text": combined_prompt}],
                "role": "user"
            }]
        })

        task_metadata.append({
            "row_index": item['row_index'],
            "style": item['style'],
            "safe_style": item['safe_style'],
            "prompt": item['prompt'],
            "combined_prompt": combined_prompt,
            "output_filename": filename
        })

    # Create batch job
    try:
        batch_job = worker_client.batches.create(
            model="models/gemini-2.5-flash-image",
            src=batch_requests,
            config={
                "display_name": f"nano-banana-batch-{batch_num}",
            },
        )

        print(f"[Worker {batch_num}] Created batch job: {batch_job.name}")
        print(f"[Worker {batch_num}] Waiting for batch to complete...")

        # Poll for completion
        count = 0
        while True:
            batch_status = worker_client.batches.get(name=batch_job.name)
            print(f"[Worker {batch_num}] Status: {batch_status.state.name} ({count})")
            count += 1

            if batch_status.state.name in ["JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED"]:
                break

            time.sleep(10)

        # Process results
        success_count = 0
        error_count = 0

        if batch_status.state.name == "JOB_STATE_SUCCEEDED":
            if batch_status.dest and batch_status.dest.inlined_responses:
                for i, inline_response in enumerate(batch_status.dest.inlined_responses):
                    task = task_metadata[i]

                    if inline_response.response:
                        try:
                            for part in inline_response.response.parts:
                                if part.inline_data:
                                    image_path = os.path.join(output_dir, task['output_filename'])
                                    image = part.as_image()
                                    image.save(image_path)

                                    # Save prompt info
                                    prompt_file = os.path.join(output_dir, f"{task['output_filename']}.prompt.txt")
                                    with open(prompt_file, 'w', encoding='utf-8') as f:
                                        f.write(f"Style: {task['style']}\n")
                                        f.write(f"Prompt: {task['prompt']}\n")
                                        f.write(f"Combined: {task['combined_prompt']}\n")
                                        f.write(f"Row Index: {task['row_index']}\n")
                                        f.write(f"Filename: {task['output_filename']}\n")

                                    success_count += 1
                                    break
                        except Exception as e:
                            error_count += 1
                    else:
                        error_count += 1

        return {
            'batch_num': batch_num,
            'success': success_count,
            'errors': error_count,
            'total': len(batch_requests)
        }

    except Exception as e:
        print(f"[Worker {batch_num}] Error: {e}")
        return {
            'batch_num': batch_num,
            'success': 0,
            'errors': len(batch_requests),
            'total': len(batch_requests),
            'error': str(e)
        }


def process_batch(batch: list, output_dir: str, batch_num: int, total_batches: int):
    """Process a single batch using Gemini batch API"""
    print(f"\n{'='*60}")
    print(f"Processing Batch {batch_num}/{total_batches} ({len(batch)} images)")
    print(f"{'='*60}\n")

    # Prepare batch requests
    batch_requests = []
    task_metadata = []

    for item in batch:
        # Combine style and prompt
        combined_prompt = f"Style: {item['style']}. {item['prompt']}"

        # Create filename with row index for uniqueness
        filename = f"row_{item['row_index']}_{item['safe_style']}.png"

        batch_requests.append({
            "contents": [{
                "parts": [{"text": combined_prompt}],
                "role": "user"
            }]
        })

        task_metadata.append({
            "row_index": item['row_index'],
            "style": item['style'],
            "safe_style": item['safe_style'],
            "prompt": item['prompt'],
            "combined_prompt": combined_prompt,
            "output_filename": filename
        })

    # Create batch job with Gemini 2.5 Flash
    print(f"Creating batch job with Gemini API (50% discount)...")
    try:
        batch_job = client.batches.create(
            model="models/gemini-2.5-flash-image",
            src=batch_requests,
            config={
                "display_name": f"nano-banana-batch-{batch_num}",
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
                    style = task['style']
                    row_num = task['row_index']

                    print(f"  Processing row {row_num}: {style[:30]}...")

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
                                    f.write(f"Style: {style}\n")
                                    f.write(f"Prompt: {task['prompt']}\n")
                                    f.write(f"Combined: {task['combined_prompt']}\n")
                                    f.write(f"Row Index: {row_num}\n")
                                    f.write(f"Filename: {task['output_filename']}\n")

                                print(f"    ✓ Saved: {task['output_filename']}")
                                success_count += 1
                            else:
                                # No image data
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


def save_results(output_dir: str, all_results: list, total_prompts: int):
    """Save generation results to a file"""
    results_file = os.path.join(output_dir, "generation_results.txt")

    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("NANO BANANA BATCH GENERATION RESULTS\n")
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"CSV File: {CSV_FILE}\n")
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
        f.write(f"\n💰 Cost savings: 50% discount applied via Gemini batch mode!\n")

    print(f"\n✓ Results saved to: {results_file}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Generate images in batches using Gemini Batch API (Nano Banana)'
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
        '--csv',
        type=str,
        default=CSV_FILE,
        help=f'CSV file to read (default: {CSV_FILE})'
    )
    parser.add_argument(
        '--count',
        type=int,
        help='Number of images to generate (e.g., 10, 50, 100). Overrides --batch option.'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Number of parallel workers for batch processing (default: 1). Use 0 for auto-detect.'
    )

    args = parser.parse_args()

    # Validate batch size
    if args.batch_size > 50:
        print("WARNING: Gemini batch API has a limit of 50 requests per batch.")
        print(f"Setting batch size to 50.")
        args.batch_size = 50

    # Read prompts from CSV
    print(f"Reading prompts from: {args.csv}")
    prompts = read_csv_prompts(args.csv)

    if not prompts:
        print("ERROR: No prompts found in CSV file!")
        sys.exit(1)

    print(f"Found {len(prompts)} prompt pairs")

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
        list_batches(prompts, args.batch_size)
        return

    # Get batches
    batches = get_batches(prompts, args.batch_size)
    total_batches = len(batches)

    # Determine which batches to process
    if args.count:
        # When --count is used, process all batches from the limited prompts
        batches_to_process = list(range(total_batches))
    elif args.batch:
        if args.batch.lower() == 'all':
            batches_to_process = list(range(total_batches))
        else:
            try:
                batch_num = int(args.batch)
                if batch_num < 1 or batch_num > total_batches:
                    print(f"ERROR: Batch number must be between 1 and {total_batches}")
                    list_batches(prompts, args.batch_size)
                    sys.exit(1)
                batches_to_process = [batch_num - 1]  # Convert to 0-indexed
            except ValueError:
                print(f"ERROR: Invalid batch number '{args.batch}'. Use a number or 'all'")
                sys.exit(1)
    else:
        # No batch/count specified, ask user
        total_batches = list_batches(prompts, args.batch_size)
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
    output_dir = create_output_dir()
    print(f"\nOutput directory: {output_dir}")

    # Determine number of workers
    if args.workers == 0:
        num_workers = min(multiprocessing.cpu_count(), len(batches_to_process))
        print(f"Auto-detected {num_workers} CPU cores, using {num_workers} workers")
    else:
        num_workers = min(args.workers, len(batches_to_process))
        print(f"Using {num_workers} parallel workers")

    # Process batches
    start_time = time.time()
    all_results = []

    if num_workers > 1:
        # Parallel processing
        print(f"\nProcessing {len(batches_to_process)} batches in parallel...\n")

        # Prepare arguments for workers
        api_key = os.environ.get("GEMINI_API_KEY")
        worker_args = [
            (batches[batch_idx], output_dir, batch_idx + 1, total_batches, api_key)
            for batch_idx in batches_to_process
        ]

        # Process in parallel
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(worker_process_batch, arg): arg[2] for arg in worker_args}

            for future in as_completed(futures):
                batch_num = futures[future]
                try:
                    result = future.result()
                    all_results.append(result)
                except Exception as e:
                    print(f"\n[Worker {batch_num}] Exception: {e}")
                    all_results.append({
                        'batch_num': batch_num,
                        'success': 0,
                        'errors': 1,
                        'total': 1,
                        'error': str(e)
                    })

        # Sort results by batch number
        all_results.sort(key=lambda x: x['batch_num'])

    else:
        # Serial processing (original behavior)
        for batch_idx in batches_to_process:
            batch = batches[batch_idx]
            batch_num = batch_idx + 1

            result = process_batch(batch, output_dir, batch_num, total_batches)
            all_results.append(result)

    # Save results
    save_results(output_dir, all_results, len(prompts))

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
    print(f"\n💰 Cost savings: 50% discount applied via Gemini batch mode!")


if __name__ == "__main__":
    main()
