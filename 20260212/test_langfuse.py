#!/usr/bin/env python3
"""Test Langfuse integration"""

import os
from dotenv import load_dotenv
from langfuse import Langfuse
from datetime import datetime

load_dotenv()

# Initialize Langfuse client
langfuse = Langfuse(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host=os.getenv("LANGFUSE_BASE_URL")
)

# Test creating a generation observation
try:
    generation = langfuse.start_observation(
        name="test_video_generation",
        as_type="generation",
        model="veo-3.1-generate-001",
        model_parameters={
            "aspect_ratio": "16:9",
            "duration_seconds": 4,
            "number_of_videos": 1
        },
        input={
            "prompt": "Test prompt",
            "reference_image": "test.png"
        },
        output={
            "status": "success",
            "video_generated": True
        },
        metadata={
            "scene": "1",
            "shot_type": "A-ROLL",
            "shot_title": "Test",
            "timestamp": datetime.now().isoformat()
        }
    )
    print("[OK] Generation created successfully")

    generation.end()
    print("[OK] Generation ended successfully")

    # Flush to ensure data is sent
    langfuse.flush()
    print("[OK] Data flushed to Langfuse")

    print("\n[SUCCESS] Langfuse integration is working correctly!")

except Exception as e:
    print(f"[ERROR] {e}")
    import traceback
    traceback.print_exc()
