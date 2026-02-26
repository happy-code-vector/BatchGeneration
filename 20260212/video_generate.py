#!/usr/bin/env python3
"""
Video Generation Script using Vertex AI REST API

This script generates videos using Vertex AI API with prompts from CSV files
and reference images from Brooklyn or Hoover directories.

Usage:
    python video_generate.py --csv BROOKLYN_BRIDGE_API_Ready.csv --count 5
    python video_generate.py --csv HOOVER_DAM_API_Ready.csv --count 10
"""

import os
import argparse
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import requests
import json
import base64
import time
import re
from langfuse import Langfuse
from datetime import datetime

# Load environment variables
load_dotenv()

# Initialize Langfuse client
langfuse = Langfuse(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host=os.getenv("LANGFUSE_BASE_URL")
)

VERTEX_API_KEY = os.getenv("VERTEX_API_KEY")


def sanitize_filename(name: str) -> str:
    """Create safe filename from string (same logic as batch_generate.py)"""
    # Remove or replace unsafe characters
    safe = re.sub(r'[<>:"/\\|?*]', '', name)
    # Replace multiple spaces/hyphens with single hyphen
    safe = re.sub(r'[\s_]+', '-', safe)
    safe = re.sub(r'-+', '-', safe)
    # Remove leading/trailing hyphens
    safe = safe.strip('-')
    return safe


def get_image_directory(csv_filename):
    """Determine the image directory based on CSV filename."""
    csv_name = Path(csv_filename).stem.lower()
    if "brooklyn" in csv_name:
        return "Brooklyn"
    elif "hoover" in csv_name:
        return "Hoover"
    else:
        raise ValueError(f"Cannot determine image directory for CSV: {csv_filename}")


def find_reference_image(image_dir, scene, shot_type, shot_title):
    """
    Find the reference image based on scene, shot type, and shot title.
    Uses the same naming logic as batch_generate.py.
    Example: 5 - B - Caisson-Construction-Begins-Timber-Joint.png
    """
    # Extract first letter of shot type (same as batch_generate.py line 78)
    shot_type_letter = shot_type[0].upper() if shot_type else 'U'

    # Sanitize shot title using the same function as batch_generate.py
    safe_title = sanitize_filename(shot_title)

    # Build expected filename (same format as batch_generate.py line 82)
    expected_filename = f"{scene} - {shot_type_letter} - {safe_title}.png"
    expected_path = Path(image_dir) / expected_filename

    if expected_path.exists():
        return str(expected_path)

    # Try case-insensitive search
    image_dir_path = Path(image_dir)
    scene_lower = str(scene).lower()
    shot_type_lower = shot_type_letter.lower()
    safe_title_lower = safe_title.lower()

    for img_file in image_dir_path.glob("*.png"):
        img_name = img_file.stem.lower()

        # Build expected pattern in lowercase
        expected_pattern = f"{scene_lower} - {shot_type_lower} - {safe_title_lower}"

        if expected_pattern == img_name:
            return str(img_file)

    # List available files for debugging
    available_files = [f.name for f in image_dir_path.glob("*.png")]
    raise FileNotFoundError(
        f"Reference image not found for Scene {scene}, {shot_type}, {shot_title}\n"
        f"Expected filename: {expected_filename}\n"
        f"Shot type letter: {shot_type_letter}\n"
        f"Sanitized title: {safe_title}\n"
        f"Directory: {image_dir}\n"
        f"Available files (first 10): {available_files[:10]}"
    )


def encode_image_to_base64(image_path):
    """Encode an image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def generate_video(prompt, reference_image_path, output_path, scene, shot_type, shot_title, project_id="your-project-id", location="us-central1"):
    """
    Generate a video using Vertex AI REST API with the given prompt and reference image.

    Args:
        prompt: The text prompt for video generation
        reference_image_path: Path to the reference image
        output_path: Path where the generated video will be saved
        scene: Scene number for tracking
        shot_type: Shot type for tracking
        shot_title: Shot title for tracking
        project_id: Google Cloud project ID
        location: Vertex AI location
    """
    print(f"Generating video with prompt: {prompt[:100]}...")
    print(f"Reference image: {reference_image_path}")
    print(f"Output: {output_path}")

    # Track the result for Langfuse
    result_data = {
        "status": "pending",
        "scene": scene,
        "shot_type": shot_type,
        "shot_title": shot_title,
        "reference_image": reference_image_path,
        "output_path": output_path
    }

    try:
        # Encode the reference image to base64
        image_base64 = encode_image_to_base64(reference_image_path)

        # Vertex AI REST API endpoint for video generation
        # Note: This is a placeholder - you'll need to update with the actual endpoint
        # when Vertex AI video generation API becomes publicly available
        api_endpoint = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/google/models/veo-3.1-generate-001:predictLongRunning"

        # Prepare the request payload
        payload = {
            "instances": [
                {
                    "prompt": prompt,
                    "referenceImage": {
                        "bytesBase64Encoded": image_base64
                    },
                    "aspectRatio": "16:9",
                    "durationSeconds": 4
                }
            ]
        }

        # Set up headers with authentication
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {VERTEX_API_KEY}"
        }

        print(f"Sending request to Vertex AI API...")
        print(f"Note: Vertex AI video generation API may be in limited preview.")
        print(f"If you get authentication errors, ensure your API key has access to video generation.")

        # Make the API request
        response = requests.post(api_endpoint, json=payload, headers=headers, timeout=300)

        # Check if the request was successful
        if response.status_code == 200:
            result = response.json()

            # Handle long-running operation
            if 'name' in result:
                operation_name = result['name']
                print(f"Operation started: {operation_name}")
                print(f"Waiting for video generation to complete...")

                # Poll for operation completion
                max_attempts = 60  # 5 minutes maximum
                for attempt in range(max_attempts):
                    time.sleep(5)  # Wait 5 seconds between polls

                    poll_response = requests.get(
                        f"https://{location}-aiplatform.googleapis.com/v1/{operation_name}",
                        headers={"Authorization": f"Bearer {VERTEX_API_KEY}"}
                    )

                    if poll_response.status_code == 200:
                        poll_result = poll_response.json()

                        if 'done' in poll_result and poll_result['done']:
                            if 'response' in poll_result:
                                # Extract video data from response
                                video_data = poll_result['response']

                                # Save the video file
                                if 'videoBytes' in video_data:
                                    video_bytes = base64.b64decode(video_data['videoBytes'])
                                    with open(output_path, 'wb') as f:
                                        f.write(video_bytes)
                                    print(f"Video saved successfully to: {output_path}")
                                elif 'gcsUri' in video_data:
                                    # If video is stored in GCS, download it
                                    print(f"Video available at: {video_data['gcsUri']}")
                                    print(f"Please download manually from GCS")
                                else:
                                    print(f"Video generation complete. Response: {json.dumps(video_data, indent=2)}")

                                # Track successful result
                                result_data.update({
                                    "status": "success",
                                    "video_generated": True,
                                    "response": video_data
                                })

                                # Create Langfuse observation with result
                                langfuse.start_observation(
                                    name="vertex_ai_video_generation",
                                    as_type="generation",
                                    model="veo-3.1-generate-001",
                                    model_parameters={
                                        "aspect_ratio": "16:9",
                                        "duration_seconds": 4,
                                        "number_of_videos": 1
                                    },
                                    input={
                                        "prompt": prompt,
                                        "reference_image": reference_image_path
                                    },
                                    output=result_data,
                                    metadata={
                                        "scene": scene,
                                        "shot_type": shot_type,
                                        "shot_title": shot_title,
                                        "project_id": project_id,
                                        "location": location,
                                        "timestamp": datetime.now().isoformat()
                                    }
                                ).end()

                                return output_path
                            elif 'error' in poll_result:
                                raise Exception(f"API Error: {poll_result['error']}")
                    else:
                        print(f"Polling attempt {attempt + 1} failed: {poll_response.status_code}")

                raise Exception("Operation timed out")

            elif 'predictions' in result:
                # Handle synchronous response
                prediction = result['predictions'][0]

                # Save the video file
                if 'bytesBase64Encoded' in prediction:
                    video_bytes = base64.b64decode(prediction['bytesBase64Encoded'])
                    with open(output_path, 'wb') as f:
                        f.write(video_bytes)
                    print(f"Video saved successfully to: {output_path}")
                else:
                    print(f"Video generation complete. Response: {json.dumps(prediction, indent=2)}")

                # Track successful result
                result_data.update({
                    "status": "success",
                    "video_generated": True,
                    "response": prediction
                })

                # Create Langfuse observation with result
                langfuse.start_observation(
                    name="vertex_ai_video_generation",
                    as_type="generation",
                    model="veo-3.1-generate-001",
                    model_parameters={
                        "aspect_ratio": "16:9",
                        "duration_seconds": 4,
                        "number_of_videos": 1
                    },
                    input={
                        "prompt": prompt,
                        "reference_image": reference_image_path
                    },
                    output=result_data,
                    metadata={
                        "scene": scene,
                        "shot_type": shot_type,
                        "shot_title": shot_title,
                        "project_id": project_id,
                        "location": location,
                        "timestamp": datetime.now().isoformat()
                    }
                ).end()

                return output_path
            else:
                print(f"Unexpected response format: {json.dumps(result, indent=2)}")
                raise Exception("Unexpected response format from API")
        else:
            error_msg = f"API request failed with status {response.status_code}: {response.text}"
            print(f"Error: {error_msg}")

            # Track error result
            result_data.update({
                "status": "error",
                "error_message": error_msg,
                "video_generated": False,
                "status_code": response.status_code,
                "response_text": response.text
            })

            # Create Langfuse observation with error
            langfuse.start_observation(
                name="vertex_ai_video_generation",
                as_type="generation",
                model="veo-3.1-generate-001",
                model_parameters={
                    "aspect_ratio": "16:9",
                    "duration_seconds": 4,
                    "number_of_videos": 1
                },
                input={
                    "prompt": prompt,
                    "reference_image": reference_image_path
                },
                output=result_data,
                level="ERROR",
                metadata={
                    "scene": scene,
                    "shot_type": shot_type,
                    "shot_title": shot_title,
                    "project_id": project_id,
                    "location": location,
                    "timestamp": datetime.now().isoformat()
                }
            ).end()

            raise Exception(error_msg)

    except Exception as e:
        # Track error result
        result_data.update({
            "status": "error",
            "error_message": str(e),
            "video_generated": False
        })

        # Create Langfuse observation with error
        langfuse.start_observation(
            name="vertex_ai_video_generation",
            as_type="generation",
            model="veo-3.1-generate-001",
            model_parameters={
                "aspect_ratio": "16:9",
                "duration_seconds": 4,
                "number_of_videos": 1
            },
            input={
                "prompt": prompt,
                "reference_image": reference_image_path
            },
            output=result_data,
            level="ERROR",
            metadata={
                "scene": scene,
                "shot_type": shot_type,
                "shot_title": shot_title,
                "project_id": project_id,
                "location": location,
                "timestamp": datetime.now().isoformat()
            }
        ).end()

        raise e


def main():
    parser = argparse.ArgumentParser(
        description="Generate videos using Vertex AI API with prompts from CSV"
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to the CSV file containing prompts (e.g., BROOKLYN_BRIDGE_API_Ready.csv)"
    )
    parser.add_argument(
        "--count",
        type=int,
        required=True,
        help="Number of videos to generate"
    )
    parser.add_argument(
        "--project-id",
        type=str,
        default="your-project-id",
        help="Google Cloud project ID (default: your-project-id)"
    )
    parser.add_argument(
        "--location",
        type=str,
        default="us-central1",
        help="Vertex AI location (default: us-central1)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="generated_videos",
        help="Output directory for generated videos (default: generated_videos)"
    )

    args = parser.parse_args()

    # Validate CSV file exists
    if not Path(args.csv).exists():
        print(f"Error: CSV file not found: {args.csv}")
        return 1

    # Get the image directory
    try:
        image_dir = get_image_directory(args.csv)
        print(f"Using image directory: {image_dir}")
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    # Validate image directory exists
    if not Path(image_dir).exists():
        print(f"Error: Image directory not found: {image_dir}")
        return 1

    # Read CSV file with encoding fallback
    print(f"Reading prompts from: {args.csv}")
    try:
        df = pd.read_csv(args.csv, encoding='utf-8')
    except UnicodeDecodeError:
        print("UTF-8 encoding failed, trying latin-1...")
        df = pd.read_csv(args.csv, encoding='latin-1')

    # Validate count
    if args.count > len(df):
        print(f"Warning: Requested {args.count} videos but CSV only has {len(df)} rows")
        args.count = len(df)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Print API information
    print(f"\n{'='*60}")
    print("IMPORTANT: Vertex AI Video Generation API")
    print(f"{'='*60}")
    print("The Vertex AI video generation API (Veo model) is currently in")
    print("limited preview and may not be publicly available yet.")
    print("")
    print("If you receive authentication or endpoint errors:")
    print("1. Ensure your Google Cloud project has access to Veo API")
    print("2. Check that your API key has the necessary permissions")
    print("3. Verify the model name and endpoint are correct")
    print("4. Contact Google Cloud support for access to video generation")
    print(f"{'='*60}\n")

    # Generate videos
    print(f"\nStarting video generation for {args.count} videos...\n")

    success_count = 0
    error_count = 0

    for i in range(min(args.count, len(df))):
        row = df.iloc[i]

        scene = str(row['Scene']).strip()
        shot_type = str(row['Shot_Type']).strip()
        shot_title = str(row['Shot_Title']).strip()
        prompt = str(row['Full_Prompt (Copy & Paste Ready)']).strip()

        print(f"\n{'='*60}")
        print(f"Video {i+1}/{args.count}")
        print(f"Scene: {scene}")
        print(f"Shot Type: {shot_type}")
        print(f"Shot Title: {shot_title}")
        print(f"{'='*60}")

        try:
            # Find reference image
            ref_image = find_reference_image(image_dir, scene, shot_type, shot_title)
            print(f"Found reference image: {ref_image}")

            # Generate output filename
            safe_shot_title = shot_title.replace(" ", "-").replace("/", "-")
            output_filename = f"{scene}_{shot_type}_{safe_shot_title}.mp4"
            output_path = output_dir / output_filename

            # Check if video already exists
            if output_path.exists():
                print(f"Video already exists, skipping: {output_path}")
                continue

            # Generate video
            generate_video(
                prompt=prompt,
                reference_image_path=ref_image,
                output_path=str(output_path),
                scene=scene,
                shot_type=shot_type,
                shot_title=shot_title,
                project_id=args.project_id,
                location=args.location
            )

            print(f"✓ Successfully generated: {output_filename}")
            success_count += 1

        except FileNotFoundError as e:
            print(f"✗ Error: {e}")
            error_count += 1
            continue
        except Exception as e:
            print(f"✗ Error generating video: {e}")
            error_count += 1
            continue

    print(f"\n{'='*60}")
    print("Video generation complete!")
    print(f"Successfully generated: {success_count} videos")
    print(f"Errors: {error_count}")
    print(f"Videos saved to: {output_dir}")
    print(f"{'='*60}\n")

    # Flush Langfuse data
    langfuse.flush()

    return 0


if __name__ == "__main__":
    exit(main())
