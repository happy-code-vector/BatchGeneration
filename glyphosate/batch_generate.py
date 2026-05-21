"""
Batch Image Generation using Gemini Batch API (50% discount)
Reads from XLSX file with paired Clean/Glyphosate prompts
Outputs organized by category subdirectories
"""

import os
import sys
import re
import time
import datetime
import argparse
from pathlib import Path
from dotenv import load_dotenv

import openpyxl

load_dotenv()

BATCH_SIZE = 50

if not os.environ.get("GEMINI_API_KEY"):
    print("ERROR: GEMINI_API_KEY environment variable not set!")
    print("Please set it in .env file: GEMINI_API_KEY=your_api_key_here")
    sys.exit(1)

from google import genai

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


def sanitize_filename(name: str) -> str:
    safe = re.sub(r'[<>:"/\\|?*]', '', name)
    safe = re.sub(r'[\s_]+', '-', safe)
    safe = re.sub(r'-+', '-', safe)
    safe = safe.strip('-')
    return safe.lower()


def read_xlsx_prompts(xlsx_file: str) -> list:
    wb = openpyxl.load_workbook(xlsx_file, read_only=True)
    ws = wb["Pairs (Side by Side)"]

    prompts = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        row_id, category, food_item, clean_prompt, glyphosate_prompt, *_ = row
        if not row_id or not food_item:
            continue

        category = str(category).strip()
        food_item = str(food_item).strip()
        safe_food = sanitize_filename(food_item)
        safe_category = sanitize_filename(category)

        # Quality suffix appended to every prompt
        quality_suffix = (
            "\n\nImage specifications: 9:16 aspect ratio (vertical/portrait orientation), "
            "2K resolution (1440x2560), highest quality, photorealistic, sharp details."
        )

        if clean_prompt and str(clean_prompt).strip():
            clean_text = str(clean_prompt).strip() + quality_suffix
            prompts.append({
                "id": int(row_id),
                "category": category,
                "safe_category": safe_category,
                "food_item": food_item,
                "safe_food": safe_food,
                "prompt": clean_text,
                "label": "clean",
                "output_filename": f"{row_id}-{safe_food}-clean.png",
            })

        if glyphosate_prompt and str(glyphosate_prompt).strip():
            glyph_text = str(glyphosate_prompt).strip() + quality_suffix
            prompts.append({
                "id": int(row_id),
                "category": category,
                "safe_category": safe_category,
                "food_item": food_item,
                "safe_food": safe_food,
                "prompt": glyph_text,
                "label": "glyphosate",
                "output_filename": f"{row_id}-{safe_food}-glyphosate.png",
            })

    wb.close()
    return prompts


def create_output_dir() -> str:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"generated_images_{timestamp}"
    Path(output_dir).mkdir(exist_ok=True)
    return output_dir


def get_batches(prompts: list, batch_size: int) -> list:
    return [prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]


def list_batches(prompts: list, batch_size: int, source_file: str):
    batches = get_batches(prompts, batch_size)
    print(f"\n{'='*60}")
    print(f"Source: {source_file}")
    print(f"Total images to generate: {len(prompts)} ({len(prompts)//2} pairs)")
    print(f"Batch size: {batch_size}")
    print(f"Total batches: {len(batches)}")
    print(f"{'='*60}\n")

    for i, batch in enumerate(batches, start=1):
        categories_in_batch = set(p["safe_category"] for p in batch)
        labels = [f"{p['id']}-{p['label']}" for p in batch[:4]]
        preview = ", ".join(labels)
        if len(batch) > 4:
            preview += "..."
        print(f"Batch {i}: {len(batch)} images  [{', '.join(sorted(categories_in_batch))}]")
        print(f"       {preview}")
        print()

    return len(batches)


def process_batch(batch: list, output_dir: str, batch_num: int, total_batches: int):
    print(f"\n{'='*60}")
    print(f"Processing Batch {batch_num}/{total_batches} ({len(batch)} images)")
    print(f"{'='*60}\n")

    batch_requests = []
    task_metadata = []

    for item in batch:
        batch_requests.append({
            "contents": [{
                "parts": [{"text": item["prompt"]}],
                "role": "user"
            }]
        })
        task_metadata.append(item)

    print(f"Creating batch job with Gemini API (50% discount)...")
    try:
        batch_job = client.batches.create(
            model="models/gemini-2.5-flash-image",
            src=batch_requests,
            config={
                "display_name": f"gen-batch-{batch_num}",
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
                print(f"    [{task['id']}] {task['food_item']} ({task['label']})...", end=" ")

                # Create category subdirectory
                cat_dir = os.path.join(output_dir, task["safe_category"])
                Path(cat_dir).mkdir(exist_ok=True)

                if inline_response.response:
                    try:
                        image_parts = [
                            part for part in inline_response.response.parts
                            if part.inline_data
                        ]

                        if image_parts:
                            image_path = os.path.join(cat_dir, task["output_filename"])
                            image = image_parts[0].as_image()
                            image.save(image_path)

                            # Save prompt alongside image
                            prompt_file = os.path.join(cat_dir, f"{task['output_filename']}.prompt.txt")
                            with open(prompt_file, 'w', encoding='utf-8') as f:
                                f.write(f"ID: {task['id']}\n")
                                f.write(f"Category: {task['category']}\n")
                                f.write(f"Food Item: {task['food_item']}\n")
                                f.write(f"Label: {task['label']}\n")
                                f.write(f"Filename: {task['output_filename']}\n")
                                f.write(f"\nPrompt:\n{task['prompt']}\n")

                            print(f"OK -> {task['safe_category']}/{task['output_filename']}")
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


def save_results(output_dir: str, all_results: list, total_prompts: int, source_file: str):
    results_file = os.path.join(output_dir, "generation_results.txt")
    total_success = sum(r["success"] for r in all_results)
    total_errors = sum(r["errors"] for r in all_results)

    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("BATCH IMAGE GENERATION RESULTS\n")
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Source: {source_file}\n")
        f.write(f"Total images: {total_prompts} ({total_prompts // 2} pairs)\n")
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
        description="Generate paired clean/glyphosate images using Gemini Batch API (50% discount)"
    )
    parser.add_argument(
        "--xlsx", type=str, required=True,
        help="XLSX file to read (Pairs sheet)"
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
        help="Number of images to generate (not pairs)"
    )
    parser.add_argument(
        "--label", type=str, choices=["clean", "glyphosate", "both"], default="both",
        help="Which images to generate (default: both)"
    )

    args = parser.parse_args()

    if args.batch_size > 50:
        print("WARNING: Gemini batch API limit is 50. Setting batch size to 50.")
        args.batch_size = 50

    if not os.path.exists(args.xlsx):
        print(f"ERROR: File not found: {args.xlsx}")
        sys.exit(1)

    print(f"Reading prompts from: {args.xlsx}")
    prompts = read_xlsx_prompts(args.xlsx)

    if not prompts:
        print("ERROR: No prompts found!")
        sys.exit(1)

    # Filter by label if specified
    if args.label != "both":
        prompts = [p for p in prompts if p["label"] == args.label]
        print(f"Filtered to '{args.label}' only: {len(prompts)} images")

    print(f"Found {len(prompts)} images ({len(prompts)//2} pairs) across "
          f"{len(set(p['category'] for p in prompts))} categories")

    if args.count:
        args.count = min(args.count, len(prompts))
        prompts = prompts[:args.count]
        print(f"Limited to first {args.count} images")

    if args.list:
        list_batches(prompts, args.batch_size, args.xlsx)
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
                    list_batches(prompts, args.batch_size, args.xlsx)
                    sys.exit(1)
                batches_to_process = [batch_num - 1]
            except ValueError:
                print(f"ERROR: Invalid batch '{args.batch}'")
                sys.exit(1)
    else:
        list_batches(prompts, args.batch_size, args.xlsx)
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

    save_results(output_dir, all_results, len(prompts), args.xlsx)

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
