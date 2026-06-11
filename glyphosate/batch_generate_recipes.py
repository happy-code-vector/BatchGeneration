"""
Batch Image Generation for Recipe CSVs using Gemini Batch API (50% discount)
Reads from image_prompts_recipes_*.csv files
Outputs images named by recipe_id (e.g. 1.png, 2.png, ...)
"""

import os
import sys
import re
import csv
import time
import datetime
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BATCH_SIZE = 50

if not os.environ.get("GEMINI_API_KEY"):
    print("ERROR: GEMINI_API_KEY environment variable not set!")
    print("Please set it in .env file: GEMINI_API_KEY=your_api_key_here")
    sys.exit(1)

from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# Image generation config: 1:1 square (matches the prompt's --ar 1:1)
IMAGE_CONFIG = types.GenerateContentConfig(
    responseModalities=["IMAGE"],
    imageConfig=types.ImageConfig(
        aspectRatio="1:1",
    ),
)


def read_csv_prompts(csv_files: list) -> list:
    """Read prompts from one or more CSV files with columns: recipe_id, recipe_name, image_prompt"""
    prompts = []
    for csv_file in csv_files:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                recipe_id = row.get('recipe_id', '').strip()
                recipe_name = row.get('recipe_name', '').strip()
                image_prompt = row.get('image_prompt', '').strip()

                if not recipe_id or not image_prompt:
                    continue

                prompts.append({
                    "id": int(recipe_id),
                    "recipe_name": recipe_name,
                    "prompt": image_prompt,
                    "output_filename": f"{recipe_id}.png",
                    "source_file": os.path.basename(csv_file),
                })

    # Sort by recipe_id to ensure consistent ordering
    prompts.sort(key=lambda p: p["id"])
    return prompts


def discover_csv_files() -> list:
    """Auto-discover all image_prompts_recipes_*.csv in the current directory"""
    pattern = "image_prompts_recipes_*.csv"
    files = sorted(Path(".").glob(pattern))
    return [str(f) for f in files]


def create_output_dir() -> str:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"generated_images_recipes_{timestamp}"
    Path(output_dir).mkdir(exist_ok=True)
    return output_dir


def get_batches(prompts: list, batch_size: int) -> list:
    return [prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]


def list_batches(prompts: list, batch_size: int):
    batches = get_batches(prompts, batch_size)
    print(f"\n{'='*60}")
    print(f"Total images to generate: {len(prompts)}")
    print(f"Batch size: {batch_size}")
    print(f"Total batches: {len(batches)}")
    print(f"{'='*60}\n")

    for i, batch in enumerate(batches, start=1):
        ids = [p['id'] for p in batch[:4]]
        preview = ", ".join(str(x) for x in ids)
        if len(batch) > 4:
            preview += f" ... (+{len(batch)-4} more)"
        print(f"Batch {i}: {len(batch)} images  [IDs: {preview}]")
    print()


def process_batch(batch: list, output_dir: str, batch_num: int, total_batches: int):
    print(f"\n{'='*60}")
    print(f"Processing Batch {batch_num}/{total_batches} ({len(batch)} images)")
    print(f"{'='*60}\n")

    batch_requests = []
    task_metadata = []

    for item in batch:
        batch_requests.append(types.InlinedRequest(
            contents=[{"parts": [{"text": item["prompt"]}], "role": "user"}],
            config=IMAGE_CONFIG,
        ))
        task_metadata.append(item)

    print(f"Creating batch job with Gemini API (50% discount)...")
    try:
        batch_job = client.batches.create(
            model="models/gemini-2.5-flash-image",
            src=batch_requests,
            config={
                "display_name": f"recipes-batch-{batch_num}",
            },
        )

        print(f"  Created batch job: {batch_job.name}")
        print(f"  Status: {batch_job.state}")
        print(f"  Waiting for completion...")

        count = 0
        while True:
            batch_status = client.batches.get(name=batch_job.name)
            state = batch_status.state.name
            count += 1
            if count % 6 == 1:
                print(f"  Status: {state} (poll #{count})")

            if state in ["JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED"]:
                break

            time.sleep(10)

        if batch_status.state.name != "JOB_STATE_SUCCEEDED":
            print(f"\n  Batch failed: {batch_status.state.name}")
            return {"batch_num": batch_num, "success": 0, "errors": len(batch_requests), "total": len(batch_requests)}

        print(f"\n  Batch completed!")

        success_count = 0
        error_count = 0

        if batch_status.dest and batch_status.dest.inlined_responses:
            print("  Processing inline results...")

            for i, inline_response in enumerate(batch_status.dest.inlined_responses):
                task = task_metadata[i]
                print(f"    [{task['id']}] {task['recipe_name']}...", end=" ")

                if inline_response.response:
                    try:
                        image_parts = [
                            part for part in inline_response.response.parts
                            if part.inline_data
                        ]

                        if image_parts:
                            image_path = os.path.join(output_dir, task["output_filename"])
                            image = image_parts[0].as_image()
                            image.save(image_path)

                            # Save prompt alongside image
                            prompt_file = os.path.join(output_dir, f"{task['id']}.prompt.txt")
                            with open(prompt_file, 'w', encoding='utf-8') as f:
                                f.write(f"Recipe ID: {task['id']}\n")
                                f.write(f"Recipe Name: {task['recipe_name']}\n")
                                f.write(f"Filename: {task['output_filename']}\n")
                                f.write(f"Source: {task['source_file']}\n")
                                f.write(f"\nPrompt:\n{task['prompt']}\n")

                            print(f"OK -> {task['output_filename']}")
                            success_count += 1
                        else:
                            print("NO IMAGE DATA")
                            error_count += 1

                    except Exception as e:
                        print(f"ERROR: {e}")
                        error_count += 1
                elif inline_response.error:
                    print(f"API ERROR: {inline_response.error}")
                    error_count += 1
                else:
                    print("NO RESPONSE")
                    error_count += 1

        elif batch_status.dest and batch_status.dest.file_name:
            print(f"  Results in file: {batch_status.dest.file_name}")
            file_content = client.files.download(file=batch_status.dest.file_name)
            output_path = os.path.join(output_dir, f"batch_{batch_num}_results.json")
            with open(output_path, 'wb') as f:
                f.write(file_content)
            print(f"  Downloaded to: {output_path}")
            success_count = len(batch_requests)
        else:
            print("  No results found.")
            error_count = len(batch_requests)

        return {"batch_num": batch_num, "success": success_count, "errors": error_count, "total": len(batch_requests)}

    except Exception as e:
        print(f"\n  Error: {e}")
        return {"batch_num": batch_num, "success": 0, "errors": len(batch_requests), "total": len(batch_requests), "error": str(e)}


def save_results(output_dir: str, all_results: list, total_prompts: int, csv_files: list):
    results_file = os.path.join(output_dir, "generation_results.txt")
    total_success = sum(r["success"] for r in all_results)
    total_errors = sum(r["errors"] for r in all_results)

    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("RECIPE BATCH IMAGE GENERATION RESULTS\n")
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Sources: {', '.join(csv_files)}\n")
        f.write(f"Total images: {total_prompts}\n")
        f.write(f"{'='*60}\n\n")

        for result in all_results:
            f.write(f"Batch {result['batch_num']}: {result['success']}/{result['total']} success")
            if result.get("error"):
                f.write(f" | Error: {result['error']}")
            f.write("\n")

        f.write(f"\n{'='*60}\n")
        f.write(f"SUMMARY\n")
        f.write(f"Success: {total_success}/{total_prompts}\n")
        f.write(f"Errors: {total_errors}\n")
        f.write(f"Rate: {total_success/total_prompts*100:.1f}%\n")
        f.write(f"\n50% cost discount applied via Gemini batch mode.\n")

    print(f"\n  Results saved to: {results_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate recipe images from CSV files using Gemini Batch API (50% discount)"
    )
    parser.add_argument(
        "--csv", type=str, nargs="+",
        help="CSV file(s) to read. If not specified, auto-discovers all image_prompts_recipes_*.csv"
    )
    parser.add_argument(
        "--batch", type=str,
        help='Batch number (e.g. "1") or "all"'
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all available batches without generating"
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help=f"Images per batch (default: {BATCH_SIZE}, max: 50)"
    )
    parser.add_argument(
        "--count", type=int,
        help="Number of images to generate (from the start)"
    )
    parser.add_argument(
        "--start-id", type=int,
        help="Start from this recipe_id (inclusive)"
    )
    parser.add_argument(
        "--end-id", type=int,
        help="End at this recipe_id (inclusive)"
    )

    args = parser.parse_args()

    if args.batch_size > 50:
        print("WARNING: Gemini batch API limit is 50. Setting batch size to 50.")
        args.batch_size = 50

    # Discover or use specified CSV files
    if args.csv:
        csv_files = args.csv
        for f in csv_files:
            if not os.path.exists(f):
                print(f"ERROR: File not found: {f}")
                sys.exit(1)
    else:
        csv_files = discover_csv_files()
        if not csv_files:
            print("ERROR: No image_prompts_recipes_*.csv files found!")
            sys.exit(1)

    print(f"Reading prompts from {len(csv_files)} file(s):")
    for f in csv_files:
        print(f"  - {f}")

    prompts = read_csv_prompts(csv_files)

    if not prompts:
        print("ERROR: No prompts found!")
        sys.exit(1)

    print(f"\nLoaded {len(prompts)} recipe prompts (IDs {prompts[0]['id']}-{prompts[-1]['id']})")

    # Filter by ID range
    if args.start_id:
        prompts = [p for p in prompts if p["id"] >= args.start_id]
        print(f"Filtered to recipe_id >= {args.start_id}: {len(prompts)} remaining")
    if args.end_id:
        prompts = [p for p in prompts if p["id"] <= args.end_id]
        print(f"Filtered to recipe_id <= {args.end_id}: {len(prompts)} remaining")

    if args.count:
        args.count = min(args.count, len(prompts))
        prompts = prompts[:args.count]
        print(f"Limited to first {args.count} images")

    if args.list:
        list_batches(prompts, args.batch_size)
        return

    batches = get_batches(prompts, args.batch_size)
    total_batches = len(batches)

    if args.batch:
        if args.batch.lower() == "all":
            batches_to_process = list(range(total_batches))
        else:
            try:
                batch_num = int(args.batch)
                if batch_num < 1 or batch_num > total_batches:
                    print(f"ERROR: Batch number must be 1-{total_batches}")
                    list_batches(prompts, args.batch_size)
                    sys.exit(1)
                batches_to_process = [batch_num - 1]
            except ValueError:
                print(f"ERROR: Invalid batch '{args.batch}'")
                sys.exit(1)
    else:
        list_batches(prompts, args.batch_size)
        user_input = input(f"\nEnter batch number (1-{total_batches}), 'all', or count: ").strip()

        if user_input.lower() == "all":
            batches_to_process = list(range(total_batches))
        else:
            try:
                batch_num = int(user_input)
                if batch_num < 1 or batch_num > total_batches:
                    print(f"ERROR: Must be 1-{total_batches}")
                    sys.exit(1)
                batches_to_process = [batch_num - 1]
            except ValueError:
                try:
                    count = int(user_input)
                    prompts = prompts[:min(count, len(prompts))]
                    batches = get_batches(prompts, args.batch_size)
                    total_batches = len(batches)
                    batches_to_process = list(range(total_batches))
                    print(f"Generating first {len(prompts)} images")
                except ValueError:
                    print(f"ERROR: Invalid input '{user_input}'")
                    sys.exit(1)

    output_dir = create_output_dir()
    print(f"\nOutput directory: {output_dir}")

    start_time = time.time()
    all_results = []

    for batch_idx in batches_to_process:
        batch = batches[batch_idx]
        batch_num = batch_idx + 1
        result = process_batch(batch, output_dir, batch_num, total_batches)
        all_results.append(result)

    save_results(output_dir, all_results, len(prompts), csv_files)

    elapsed = time.time() - start_time
    total_success = sum(r["success"] for r in all_results)
    total_errors = sum(r["errors"] for r in all_results)

    print(f"\n{'='*60}")
    print(f"BATCH GENERATION COMPLETE!")
    print(f"{'='*60}")
    print(f"Total images: {len(prompts)}")
    print(f"Generated: {total_success}")
    print(f"Errors: {total_errors}")
    print(f"Time: {elapsed:.1f}s")
    print(f"Output: {output_dir}/")
    print(f"\n50% cost discount applied via Gemini batch mode!")


if __name__ == "__main__":
    main()
