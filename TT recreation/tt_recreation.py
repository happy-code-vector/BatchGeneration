"""
TikTok Slideshow Image Recreation
Downloads TikTok slideshow images, generates prompts using Gemini Vision,
and recreates images using Gemini Batch API (50% discount)
"""

import os
import sys
import time
import datetime
import argparse
import tempfile
import shutil
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image
import subprocess
import json
import re
import requests
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables
load_dotenv()

# Configuration
BATCH_SIZE = 50  # Gemini batch API limit
VISION_MODEL = "gemini-2.5-flash"
IMAGE_MODEL = "models/gemini-2.5-flash-image"

# Vision prompt template - NO TEXT in images
VISION_PROMPT_TEMPLATE = """Analyze this image and create a detailed text-to-image prompt that would recreate a similar image. Describe:
- Subject and composition
- Art style and medium (photorealistic, illustration, painting, etc.)
- Color palette and lighting
- Mood and atmosphere
- Background and setting details
- Any notable visual elements

IMPORTANT CONSTRAINTS:
- Do NOT include any text, words, letters, numbers, or typography in the generated image
- The recreation must be purely visual with no text elements whatsoever
- If the original image contains text, describe only the visual elements, not the text content

Provide only the prompt text, ready to use for image generation."""

# Check for API key
if not os.environ.get("GEMINI_API_KEY"):
    print("ERROR: GEMINI_API_KEY environment variable not set!")
    print("Please set it in .env file: GEMINI_API_KEY=your_api_key_here")
    sys.exit(1)

# Initialize Gemini client
from google import genai
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


def create_output_dirs(timestamp: str, keep_downloads: bool = False):
    """Create output directories"""
    base_dir = Path(__file__).parent

    downloads_dir = base_dir / f"downloaded_slides_{timestamp}"
    downloads_dir.mkdir(exist_ok=True)

    generated_dir = base_dir / f"generated_images_{timestamp}"
    generated_dir.mkdir(exist_ok=True)

    return downloads_dir, generated_dir


def download_tiktok_slides(url: str, output_dir: str) -> list:
    """
    Download TikTok slideshow images
    Handles both video URLs and photo mode (slideshow) URLs
    Returns list of downloaded image paths
    """
    print(f"\n{'='*60}")
    print("Downloading TikTok Slideshow")
    print(f"URL: {url}")
    print(f"{'='*60}\n")

    # Check if this is a photo mode URL
    is_photo_mode = '/photo/' in url

    if is_photo_mode:
        print("Detected TikTok Photo Mode (slideshow)")
        return download_tiktok_photo_mode(url, output_dir)
    else:
        # Try yt-dlp for regular video URLs
        return download_tiktok_video(url, output_dir)


def download_tiktok_photo_mode(url: str, output_dir: str) -> list:
    """
    Download images from TikTok Photo Mode (slideshow)
    Uses multiple methods to extract images
    """
    print("Fetching TikTok photo mode...")

    # Extract post ID and username from URL
    # URL format: https://www.tiktok.com/@username/photo/POSTID
    match = re.search(r'@([^/]+)/photo/(\d+)', url)
    if not match:
        print("Could not parse TikTok URL")
        return []

    username = match.group(1)
    post_id = match.group(2)
    print(f"Username: @{username}, Post ID: {post_id}")

    # Method 1: Try TikTok internal API
    try:
        print("\nTrying TikTok API method...")
        images = download_via_tiktok_api(url, username, post_id, output_dir)
        if images:
            return images
    except Exception as e:
        print(f"API method failed: {e}")

    # Method 2: Try Selenium with shorter timeout
    try:
        print("\nTrying browser automation method...")
        images = download_with_selenium(url, output_dir)
        if images:
            return images
    except Exception as e:
        print(f"Selenium method failed: {e}")

    # Method 3: Fallback to HTTP with different patterns
    print("\nTrying HTTP extraction method...")
    return download_with_http(url, output_dir)


def download_via_tiktok_api(original_url: str, username: str, post_id: str, output_dir: str) -> list:
    """Use TikTok's internal API to fetch post data"""

    # TikTok's internal API endpoint for post info
    api_url = f"https://www.tiktok.com/api/post/item_list/?aid=1988&count=1&secUid=&id={post_id}"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Referer': f'https://www.tiktok.com/@{username}/photo/{post_id}',
        'Accept': 'application/json',
    }

    response = requests.get(api_url, headers=headers, timeout=30)
    print(f"API Response status: {response.status_code}")

    if response.status_code == 200:
        try:
            data = response.json()
            # Save debug JSON
            debug_file = os.path.join(output_dir, "_debug_api_response.json")
            with open(debug_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"Saved API response to: {debug_file}")

            # Extract image URLs from API response
            image_urls = []

            if 'itemList' in data:
                for item in data['itemList']:
                    if 'imagePost' in item:
                        images = item['imagePost'].get('images', [])
                        for img in images:
                            if 'imageURL' in img:
                                image_urls.append(img['imageURL'])
                            elif 'urlList' in img:
                                image_urls.extend(img['urlList'])

            if image_urls:
                print(f"Found {len(image_urls)} images via API")
                return download_images_from_urls(image_urls, output_dir)

        except json.JSONDecodeError:
            print("Could not parse API response as JSON")

    # Try alternative API endpoint
    alt_api_url = f"https://www.tiktok.com/api/comment/list/?aweme_id={post_id}"

    # Try fetching the page and looking for embedded JSON data
    response = requests.get(original_url, headers=headers, timeout=30)
    if response.status_code == 200:
        # Look for image URLs in the response
        html = response.text

        # Try to find JSON data embedded in the page
        json_patterns = [
            r'"imagePost"\s*:\s*\{[^}]*"images"\s*:\s*\[([^\]]+)\]',
            r'"imageURL"\s*:\s*"([^"]+)"',
            r'"urlList"\s*:\s*\[([^\]]+)\]',
        ]

        image_urls = []
        for pattern in json_patterns:
            matches = re.findall(pattern, html)
            for match in matches:
                # Extract URLs from the match
                url_matches = re.findall(r'https://[^"\',\s]+', match)
                image_urls.extend(url_matches)

        # Also try regex extraction
        image_urls.extend(extract_images_regex(html))

        if image_urls:
            # Filter to only TikTok CDN URLs
            tiktok_urls = [u for u in image_urls if 'tiktok' in u or 'tos-' in u]

            # Remove duplicates
            seen = set()
            unique_urls = []
            for u in tiktok_urls:
                if u not in seen and 'avatar' not in u and 'music' not in u:
                    seen.add(u)
                    unique_urls.append(u)

            if unique_urls:
                print(f"Found {len(unique_urls)} images via page extraction")
                return download_images_from_urls(unique_urls, output_dir)

    return []


def download_with_selenium(url: str, output_dir: str) -> list:
    """Use Selenium to render page and extract images"""
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from webdriver_manager.chrome import ChromeDriverManager
    import time

    print("Starting Chrome browser...")

    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920,1080')
    options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
    options.set_capability('pageLoadStrategy', 'eager')  # Don't wait for full load

    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        driver.set_page_load_timeout(60)  # Longer timeout
    except Exception as e:
        print(f"ChromeDriver setup failed: {e}")
        return []

    try:
        print("Navigating to TikTok URL...")
        try:
            driver.get(url)
        except Exception as e:
            print(f"Page load timeout (partial load): {e}")

        print("Waiting for page content to load...")
        time.sleep(5)  # Wait for initial JS

        # Scroll down to trigger lazy loading of all images
        print("Scrolling to load all images...")
        last_height = driver.execute_script("return document.body.scrollHeight")
        scroll_attempts = 0
        max_scroll_attempts = 5

        while scroll_attempts < max_scroll_attempts:
            # Scroll down
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)  # Wait for new content to load

            # Check if we've reached the bottom
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
            scroll_attempts += 1
            print(f"  Scroll {scroll_attempts}/{max_scroll_attempts}...")

        # Scroll back up to ensure all swiper slides are loaded
        driver.execute_script("window.scrollTo(0, 0);")
        time.sleep(1)

        # Try to navigate through swiper slides if they exist
        try:
            driver.execute_script("""
                // Try to trigger swiper slide loading
                const swiperSlides = document.querySelectorAll('.swiper-slide img');
                swiperSlides.forEach(img => {
                    // Force load by triggering events
                    img.dispatchEvent(new Event('load'));
                });
            """)
            time.sleep(1)
        except:
            pass

        image_urls = []

        # Extract from page source
        page_source = driver.page_source

        # Save debug HTML
        debug_file = os.path.join(output_dir, "_debug_selenium_html.txt")
        with open(debug_file, 'w', encoding='utf-8') as f:
            f.write(page_source[:200000])
        print(f"Saved debug HTML ({len(page_source)} chars)")

        # Extract image URLs using regex
        image_urls.extend(extract_images_regex(page_source))

        # Try to find img elements directly
        try:
            images = driver.find_elements(By.TAG_NAME, 'img')
            print(f"Found {len(images)} img elements")
            for img in images:
                src = img.get_attribute('src')
                if src and ('tiktokcdn' in src or 'tos-' in src) and 'avatar' not in src and 'music' not in src:
                    image_urls.append(src)

                # Also check data-src for lazy loaded images
                data_src = img.get_attribute('data-src')
                if data_src and ('tiktokcdn' in data_src or 'tos-' in data_src):
                    image_urls.append(data_src)
        except Exception as e:
            print(f"Error finding img elements: {e}")

        # Try to get swiper slide images specifically
        try:
            swiper_images = driver.find_elements(By.CSS_SELECTOR, '.swiper-slide img')
            print(f"Found {len(swiper_images)} swiper images")
            for img in swiper_images:
                src = img.get_attribute('src')
                if src and ('tiktokcdn' in src or 'tos-' in src):
                    image_urls.append(src)
        except:
            pass

        # Remove duplicates
        seen = set()
        unique_urls = []
        for u in image_urls:
            # Clean HTML entities
            u = u.replace('&amp;', '&')
            if u not in seen:
                seen.add(u)
                unique_urls.append(u)

        print(f"Found {len(unique_urls)} unique image URLs")

        if unique_urls:
            return download_images_from_urls(unique_urls, output_dir)

        return []

    finally:
        driver.quit()
        print("Browser closed.")


def download_with_http(url: str, output_dir: str) -> list:
    """Fallback HTTP request method"""
    print("Fetching via HTTP...")

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Referer': 'https://www.tiktok.com/',
    }

    try:
        response = requests.get(url, headers=headers, timeout=30)
        print(f"Response status: {response.status_code}")

        if response.status_code != 200:
            print(f"Failed to fetch page: {response.status_code}")
            return []

        html_content = response.text

        # Debug: Save HTML
        debug_file = os.path.join(output_dir, "_debug_http_html.txt")
        with open(debug_file, 'w', encoding='utf-8') as f:
            f.write(html_content[:50000])

        # Extract images
        image_urls = extract_images_from_html(html_content)
        image_urls.extend(extract_images_regex(html_content))

        # Remove duplicates
        seen = set()
        unique_urls = []
        for u in image_urls:
            if u not in seen and 'avatar' not in u and 'music' not in u:
                seen.add(u)
                unique_urls.append(u)

        if unique_urls:
            print(f"Found {len(unique_urls)} image URLs")
            return download_images_from_urls(unique_urls, output_dir)

        print("Could not extract images from page")
        return []

    except Exception as e:
        print(f"Error fetching page: {e}")
        return []


def extract_images_from_html(html: str) -> list:
    """Extract image URLs from TikTok HTML page"""
    image_urls = []

    # Look for SIGI_STATE JSON
    sigi_match = re.search(r'<script id="SIGI_STATE" type="application/json">(.+?)</script>', html)
    if sigi_match:
        try:
            data = json.loads(sigi_match.group(1))
            # Navigate to find image URLs
            if 'ItemModule' in data:
                for item_id, item in data['ItemModule'].items():
                    if 'imagePost' in item:
                        images = item['imagePost'].get('images', [])
                        for img in images:
                            if 'imageURL' in img:
                                image_urls.append(img['imageURL'])
                            elif 'urlList' in img:
                                image_urls.extend(img['urlList'])
        except json.JSONDecodeError:
            pass

    # Look for __NEXT_DATA__ JSON
    next_match = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.+?)</script>', html)
    if next_match:
        try:
            data = json.loads(next_match.group(1))
            # Navigate through props -> pageProps -> itemInfo -> itemStruct
            props = data.get('props', {})
            page_props = props.get('pageProps', {})
            item_info = page_props.get('itemInfo', {})
            item_struct = item_info.get('itemStruct', {})

            # Check for imagePost
            image_post = item_struct.get('imagePost', {})
            if image_post:
                images = image_post.get('images', [])
                for img in images:
                    # Get the highest quality URL
                    if 'urlList' in img:
                        # Usually last URL is highest quality
                        url_list = img['urlList']
                        if url_list:
                            image_urls.append(url_list[-1])
                    elif 'imageURL' in img:
                        image_urls.append(img['imageURL'])

            # Check for video slides
            video = item_struct.get('video', {})
            if video:
                # Check for playAddr which might have multiple images
                play_addr = video.get('playAddr', [])
                if isinstance(play_addr, list):
                    for addr in play_addr:
                        if 'src' in addr:
                            image_urls.append(addr['src'])
        except json.JSONDecodeError:
            pass

    return image_urls


def extract_images_regex(html: str) -> list:
    """Extract TikTok CDN image URLs using regex"""
    image_urls = []

    # Pattern for TikTok CDN images - updated for photo mode
    patterns = [
        # Photo mode images (swiper slides)
        r'https://p(?:16|19)-common-sign\.tiktokcdn-us\.com/[^"\'\s]+\.(?:jpeg|jpg|png|webp)(?:\?[^"\'\s]*)?',
        r'https://p(?:16|19)-common-sign\.tiktokcdn[^"\']+~tplv-photomode[^"\'\s]*',
        # Standard TikTok CDN patterns
        r'https://p(?:16|19|77|79)\.sign\.tiktokcdn-us\.com/[^"\'\s]+\.(?:jpeg|jpg|png|webp)',
        r'https://p(?:16|19|77|79)-sign\.tiktokcdn-us\.com/[^"\'\s]+\.(?:jpeg|jpg|png|webp)',
        r'https://v(?:16|19|77|79)-sign\.tiktokcdn\.com/[^"\'\s]+\.(?:jpeg|jpg|png|webp)',
        # TOS patterns
        r'https://tos-alisg\.be\.tiktokcdn\.com/[^"\'\s]+\.(?:jpeg|jpg|png|webp)',
        r'https://tos-useast[^"\']+\.tiktokcdn\.com/[^"\'\s]+\.(?:jpeg|jpg|png|webp)',
        # Generic tiktokcdn with photomode
        r'https://[^"\'\s]*tiktokcdn[^"\'\s]*photomode[^"\'\s]*',
        # Direct image URLs with tos- pattern
        r'https://[^"\'\s]*tiktokcdn[^"\'\s]*/tos-[^"\'\s]+\.(?:jpeg|jpg|png|webp)',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, html, re.IGNORECASE)
        image_urls.extend(matches)

    # Remove duplicates
    seen = set()
    unique_urls = []
    for url in image_urls:
        # Clean URL (remove trailing chars and HTML entities)
        url = url.rstrip('.,;:')
        url = url.replace('&amp;', '&')
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)

    return unique_urls


def download_tiktok_video(url: str, output_dir: str) -> list:
    """Download TikTok video using yt-dlp"""
    print("Attempting to download with yt-dlp...")

    # Check if yt-dlp is installed
    try:
        result = subprocess.run(["yt-dlp", "--version"], capture_output=True, text=True)
        print(f"yt-dlp version: {result.stdout.strip()}")
    except FileNotFoundError:
        print("ERROR: yt-dlp is not installed!")
        print("Install with: pip install yt-dlp")
        return []

    output_template = os.path.join(output_dir, "slide_%(autonumber)02d.%(ext)s")

    cmd = [
        "yt-dlp",
        "--no-playlist",
        "-o", output_template,
        "--no-check-certificate",
        url
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f"yt-dlp error: {result.stderr[:500]}")
    except subprocess.TimeoutExpired:
        print("ERROR: Download timed out")
    except Exception as e:
        print(f"Download error: {e}")

    # Find downloaded files
    downloaded_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.webp', '*.gif', '*.mp4']:
        downloaded_files.extend(Path(output_dir).glob(ext))

    return sorted(downloaded_files, key=lambda x: x.name)


def extract_image_urls_from_json(data: dict) -> list:
    """Extract image URLs from TikTok JSON response"""
    image_urls = []

    # TikTok photo mode images are typically in 'formats' with vcodec='none'
    if 'formats' in data:
        for fmt in data['formats']:
            # Image formats have no video codec
            if fmt.get('vcodec') == 'none' and 'url' in fmt:
                url = fmt['url']
                # Filter out small thumbnails and profile pics
                if 'avatar' not in url and 'music' not in url:
                    image_urls.append(url)

    # Check entries for slideshow
    if 'entries' in data:
        for entry in data['entries']:
            if 'url' in entry:
                image_urls.append(entry['url'])
            if 'formats' in entry:
                for fmt in entry['formats']:
                    if fmt.get('vcodec') == 'none' and 'url' in fmt:
                        image_urls.append(fmt['url'])

    # Check for thumbnails that might be full images
    if 'thumbnails' in data:
        for thumb in data['thumbnails']:
            if 'url' in thumb:
                url = thumb['url']
                # Only add if it looks like a full-size image
                if 'tiktok' in url and ('image' in url or 'photo' in url or 'tos' in url):
                    image_urls.append(url)

    # Remove duplicates while preserving order
    seen = set()
    unique_urls = []
    for url in image_urls:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)

    return unique_urls


def download_images_from_urls(urls: list, output_dir: str) -> list:
    """Download images from a list of URLs"""
    downloaded = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Referer': 'https://www.tiktok.com/'
    }

    for i, img_url in enumerate(urls, 1):
        print(f"  Downloading image {i}/{len(urls)}...")
        try:
            response = requests.get(img_url, headers=headers, timeout=30)
            if response.status_code == 200:
                # Determine extension from content-type
                content_type = response.headers.get('content-type', '')
                if 'png' in content_type:
                    ext = '.png'
                elif 'webp' in content_type:
                    ext = '.webp'
                elif 'gif' in content_type:
                    ext = '.gif'
                else:
                    ext = '.jpg'

                filename = os.path.join(output_dir, f"slide_{i:02d}{ext}")
                with open(filename, 'wb') as f:
                    f.write(response.content)

                downloaded.append(Path(filename))
                print(f"    Saved: slide_{i:02d}{ext}")
            else:
                print(f"    Failed: HTTP {response.status_code}")
        except Exception as e:
            print(f"    Failed: {e}")

    return sorted(downloaded, key=lambda x: x.name)


def download_tiktok_direct(url: str, output_dir: str) -> list:
    """Direct download method using yt-dlp"""
    print("\nTrying direct download with yt-dlp...")

    output_template = os.path.join(output_dir, "slide_%(autonumber)02d.%(ext)s")

    cmd = [
        "yt-dlp",
        "--no-playlist",
        "-o", output_template,
        "--no-check-certificate",
        url
    ]

    print(f"Running: yt-dlp --no-playlist -o {output_template} [URL]")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f"yt-dlp error: {result.stderr[:500]}")
    except subprocess.TimeoutExpired:
        print("ERROR: Download timed out")
    except Exception as e:
        print(f"Download error: {e}")

    # Find downloaded files
    downloaded_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.webp', '*.gif']:
        downloaded_files.extend(Path(output_dir).glob(ext))

    downloaded_files.sort(key=lambda x: x.name)

    if not downloaded_files:
        print("\nNo images found. Trying to download video and extract frames...")
        return download_and_extract_frames(url, output_dir)

    return downloaded_files


def download_and_extract_frames(url: str, output_dir: str) -> list:
    """Download video and extract frames as last resort"""
    print("\nAttempting to download video and extract frames...")

    video_path = os.path.join(output_dir, "video.mp4")

    cmd = [
        "yt-dlp",
        "--no-playlist",
        "-o", video_path,
        "--no-check-certificate",
        url
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f"Video download failed: {result.stderr[:300]}")
            return []
    except Exception as e:
        print(f"Video download error: {e}")
        return []

    # Check if video was downloaded
    if not os.path.exists(video_path):
        # Find any video file
        for f in Path(output_dir).glob("*.mp4"):
            video_path = str(f)
            break

    if not os.path.exists(video_path):
        print("No video file downloaded")
        return []

    print(f"Video downloaded: {video_path}")
    print("Extracting frames from video...")

    # Extract frames using ffmpeg if available
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            # Save every Nth frame (adjust as needed)
            if frame_count % 30 == 0:  # Every 30 frames (~1 second at 30fps)
                frame_path = os.path.join(output_dir, f"slide_{len(frames)+1:02d}.jpg")
                cv2.imwrite(frame_path, frame)
                frames.append(Path(frame_path))
                print(f"  Extracted frame {len(frames)}")

        cap.release()

        # Remove video file
        os.remove(video_path)

        return frames

    except ImportError:
        print("OpenCV not available, cannot extract frames from video")
        print("Install with: pip install opencv-python")
        return []


def image_to_prompt(image_path: str, slide_num: int) -> str:
    """
    Use Gemini Vision to generate a prompt from an image
    """
    print(f"  Analyzing slide {slide_num} with Gemini Vision...")

    try:
        # Read image
        image = Image.open(image_path)

        # Call Gemini Vision
        response = client.models.generate_content(
            model=VISION_MODEL,
            contents=[
                VISION_PROMPT_TEMPLATE,
                image
            ]
        )

        prompt = response.text.strip()
        print(f"    Generated prompt ({len(prompt)} chars)")
        return prompt

    except Exception as e:
        print(f"    ERROR: Vision analysis failed: {e}")
        return f"ERROR: Could not generate prompt for slide {slide_num}: {str(e)}"


def parse_skip_arg(skip_str: str) -> list:
    """
    Parse skip argument like "5" or "3,5,7" into list of integers
    """
    if not skip_str:
        return []

    skip_list = []
    for part in skip_str.split(','):
        part = part.strip()
        if part:
            try:
                skip_list.append(int(part))
            except ValueError:
                print(f"WARNING: Invalid skip value '{part}', ignoring")

    return sorted(skip_list)


def get_batches(prompts: list, batch_size: int) -> list:
    """Split prompts into batches"""
    batches = []
    for i in range(0, len(prompts), batch_size):
        batches.append(prompts[i:i + batch_size])
    return batches


def process_batch(batch: list, output_dir: str, batch_num: int, total_batches: int, total_variations: int = 1):
    """Process a single batch using Gemini batch API"""
    print(f"\n{'='*60}")
    print(f"Processing Batch {batch_num}/{total_batches} ({len(batch)} images)")
    print(f"{'='*60}\n")

    # Prepare batch requests
    batch_requests = []
    task_metadata = []

    for item in batch:
        prompt_text = item['prompt']
        variation = item.get('variation', 1)
        slide_num = item['slide_num']

        # Create subdirectory for each slide
        slide_dir = os.path.join(output_dir, f"slide_{slide_num:02d}")
        Path(slide_dir).mkdir(exist_ok=True)

        # Filename format depends on variations
        if total_variations > 1:
            filename = f"var_{variation:02d}.png"
        else:
            filename = f"slide_{slide_num:02d}.png"

        batch_requests.append({
            "contents": [{
                "parts": [{"text": prompt_text}],
                "role": "user"
            }]
        })

        task_metadata.append({
            "slide_num": slide_num,
            "variation": variation,
            "prompt": prompt_text,
            "output_filename": filename,
            "slide_dir": slide_dir,
            "original_image": item.get('original_image', '')
        })

    # Create batch job with Gemini 2.5 Flash
    print(f"Creating batch job with Gemini API (50% discount)...")
    try:
        batch_job = client.batches.create(
            model=IMAGE_MODEL,
            src=batch_requests,
            config={
                "display_name": f"tt-recreation-batch-{batch_num}",
            },
        )

        print(f"Created batch job: {batch_job.name}")
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
            print(f"\nBatch completed successfully!")

            # Process results
            success_count = 0
            error_count = 0

            if batch_status.dest and batch_status.dest.inlined_responses:
                print("Processing inline results...")

                for i, inline_response in enumerate(batch_status.dest.inlined_responses):
                    task = task_metadata[i]
                    slide_num = task['slide_num']
                    variation = task.get('variation', 1)
                    slide_dir = task.get('slide_dir', output_dir)

                    print(f"  Processing slide {slide_num}, variation {variation}...")

                    if inline_response.response:
                        try:
                            image_parts = [
                                part.inline_data.data
                                for part in inline_response.response.parts
                                if part.inline_data
                            ]

                            if image_parts:
                                image_path = os.path.join(slide_dir, task['output_filename'])

                                for part in inline_response.response.parts:
                                    if part.inline_data:
                                        image = part.as_image()
                                        image.save(image_path)
                                        break

                                # Save prompt info
                                prompt_file = os.path.join(slide_dir, f"{task['output_filename']}.prompt.txt")
                                with open(prompt_file, 'w', encoding='utf-8') as f:
                                    f.write(f"Source: TikTok Slideshow\n")
                                    f.write(f"Original Slide: {slide_num}\n")
                                    f.write(f"Variation: {variation}\n")
                                    f.write(f"Original Image: {task['original_image']}\n")
                                    f.write(f"---\n\nVision Prompt:\n{task['prompt']}\n")

                                print(f"    Saved: {os.path.basename(slide_dir)}/{task['output_filename']}")
                                success_count += 1
                            else:
                                print(f"    No image data in response")
                                error_count += 1

                        except Exception as e:
                            print(f"    Error: {e}")
                            error_count += 1
                    elif inline_response.error:
                        print(f"    API Error: {inline_response.error}")
                        error_count += 1
                    else:
                        print(f"    No response received")
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
            print(f"\nBatch failed with status: {batch_status.state.name}")
            return {
                'batch_num': batch_num,
                'success': 0,
                'errors': len(batch_requests),
                'total': len(batch_requests)
            }

    except Exception as e:
        print(f"\nError creating batch job: {e}")
        return {
            'batch_num': batch_num,
            'success': 0,
            'errors': len(batch_requests),
            'total': len(batch_requests),
            'error': str(e)
        }


def save_results(output_dir: str, all_results: list, total_prompts: int, url: str, skip_list: list):
    """Save generation results to a file"""
    results_file = os.path.join(output_dir, f"generation_results_{datetime.datetime.now().strftime('%Y%m%d')}.txt")

    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("TIKTOK SLIDESHOW RECREATION RESULTS\n")
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Source URL: {url}\n")
        f.write(f"Skipped Slides: {skip_list if skip_list else 'None'}\n")
        f.write(f"Total prompts processed: {total_prompts}\n")
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
        if total_prompts > 0:
            f.write(f"Success rate: {total_success/total_prompts*100:.1f}%\n")
        f.write(f"\nCost savings: 50% discount applied via Gemini batch mode!\n")

    print(f"\nResults saved to: {results_file}")


def parse_urls_from_file(filepath: str) -> list:
    """
    Parse URLs from a file. Handles:
    - Line by line (one URL per line)
    - Comma separated
    - Tab separated
    - Space separated

    Returns list of URLs.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"ERROR: URL file not found: {filepath}")
        return []
    except Exception as e:
        print(f"ERROR: Could not read URL file: {e}")
        return []

    # Replace tabs and newlines with spaces, then split by common delimiters
    content = content.replace('\t', ' ').replace('\n', ' ').replace(',', ' ')

    # Split by space and filter empty strings
    urls = [url.strip() for url in content.split(' ') if url.strip()]

    # Filter to only valid TikTok URLs
    tiktok_urls = []
    for url in urls:
        # Accept tiktok.com URLs
        if 'tiktok.com' in url:
            tiktok_urls.append(url)
        else:
            print(f"Warning: Skipping invalid URL: {url}")

    return tiktok_urls


def process_source(url: str, input_dir: str, skip_list: list, bulk: int,
                   output_base: str, keep: bool, variations: int,
                   list_only: bool, dry_run: bool, timestamp: str,
                   url_index: int = 0, total_urls: int = 1) -> dict:
    """
    Process a single source (URL or input directory).

    Returns dict with processing results.
    """
    source_label = f"URL {url_index}/{total_urls}" if total_urls > 1 else "Source"

    if url_index > 0 and total_urls > 1:
        print(f"\n{'#'*60}")
        print(f"# PROCESSING {source_label}")
        print(f"# URL: {url}")
        print(f"{'#'*60}\n")

    # Determine output directories
    if output_base:
        base_dir = Path(output_base)
    else:
        base_dir = Path('.')

    # Create unique subdirectory for this URL
    if url:
        url_slug = url.split('/')[-1] if '/' in url else f"url_{url_index}"
        downloads_dir = base_dir / f"downloaded_slides_{timestamp}_{url_slug}"
        generated_dir = base_dir / f"generated_images_{timestamp}_{url_slug}"
    else:
        # Using input-dir
        downloads_dir = Path(input_dir)
        if output_base:
            generated_dir = base_dir / f"generated_images_{timestamp}"
        else:
            generated_dir = Path(input_dir).parent / f"generated_images_{timestamp}"

    # Get slide images
    if input_dir:
        print(f"\n{'='*60}")
        print("Loading Images from Local Folder")
        print(f"Folder: {input_dir}")
        print(f"{'='*60}\n")

        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"ERROR: Input directory not found: {input_dir}")
            return {'success': 0, 'errors': 0, 'total': 0, 'error': 'Input directory not found'}

        # Find all image files
        downloaded_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.webp', '*.gif']:
            downloaded_files.extend(input_path.glob(ext))

        downloaded_files.sort(key=lambda x: x.name)

        if not downloaded_files:
            print(f"ERROR: No image files found in: {input_dir}")
            return {'success': 0, 'errors': 0, 'total': 0, 'error': 'No image files found'}

        print(f"Found {len(downloaded_files)} images:")
        for i, f in enumerate(downloaded_files, 1):
            print(f"  Slide {i}: {f.name}")

    else:
        # Download from TikTok URL
        generated_dir.mkdir(exist_ok=True)
        downloads_dir.mkdir(exist_ok=True)

        downloaded_files = download_tiktok_slides(url, str(downloads_dir))

        if not downloaded_files:
            print("ERROR: No slides downloaded!")
            return {'success': 0, 'errors': 0, 'total': 0, 'error': 'No slides downloaded'}

    # List mode - just show slides and exit
    if list_only:
        print(f"\n{'='*60}")
        print("SLIDE LIST")
        print(f"{'='*60}")
        for i, f in enumerate(downloaded_files, 1):
            skip_marker = " [SKIP]" if i in skip_list else ""
            print(f"  Slide {i}: {f.name}{skip_marker}")
        print(f"\nTotal slides: {len(downloaded_files)}")
        print(f"Slides to generate: {len(downloaded_files) - len([s for s in skip_list if s <= len(downloaded_files)])}")
        return {'success': 0, 'errors': 0, 'total': len(downloaded_files)}

    # Generate prompts from images (Vision API)
    print(f"\n{'='*60}")
    print("Generating Prompts from Images (Gemini Vision)")
    print(f"{'='*60}\n")

    prompts_data = []
    for i, image_path in enumerate(downloaded_files, 1):
        slide_num = i

        # Skip if in skip list
        if slide_num in skip_list:
            print(f"  Skipping slide {slide_num}")
            continue

        prompt = image_to_prompt(str(image_path), slide_num)

        # Create multiple variations if requested
        for var_num in range(1, variations + 1):
            prompts_data.append({
                'slide_num': slide_num,
                'variation': var_num,
                'prompt': prompt,
                'original_image': image_path.name
            })

        if variations > 1:
            print(f"    Created {variations} variations for slide {slide_num}")

    if not prompts_data:
        print("ERROR: No prompts generated!")
        return {'success': 0, 'errors': 0, 'total': 0, 'error': 'No prompts generated'}

    print(f"\nGenerated {len(prompts_data)} prompts")

    # Dry run mode - stop here
    if dry_run:
        print("\n[DRY RUN] Prompts generated. Skipping image creation.")
        for item in prompts_data:
            print(f"\n--- Slide {item['slide_num']} ---")
            print(item['prompt'][:500] + "..." if len(item['prompt']) > 500 else item['prompt'])
        return {'success': 0, 'errors': 0, 'total': len(prompts_data)}

    # Batch image generation
    batches = get_batches(prompts_data, bulk)
    total_batches = len(batches)

    print(f"\n{'='*60}")
    print(f"Starting Batch Image Generation")
    print(f"Total images: {len(prompts_data)}")
    print(f"Batch size: {bulk}")
    print(f"Total batches: {total_batches}")
    print(f"{'='*60}\n")

    # Process batches
    start_time = time.time()
    all_results = []

    for batch_idx, batch in enumerate(batches):
        batch_num = batch_idx + 1
        result = process_batch(batch, str(generated_dir), batch_num, total_batches, variations)
        all_results.append(result)

    # Save results
    save_results(str(generated_dir), all_results, len(prompts_data), url, skip_list)

    elapsed_time = time.time() - start_time
    total_success = sum(r['success'] for r in all_results)
    total_errors = sum(r['errors'] for r in all_results)

    print(f"\n{'='*60}")
    print(f"RECREATION COMPLETE!")
    print(f"{'='*60}")
    print(f"Source URL: {url}")
    print(f"Total images: {len(prompts_data)}")
    print(f"Generated: {total_success}")
    print(f"Errors: {total_errors}")
    print(f"Time elapsed: {elapsed_time:.1f} seconds")
    print(f"Output directory: {generated_dir}")
    print(f"\nCost savings: 50% discount applied via Gemini batch mode!")

    # Clean up downloads if not keeping
    if not keep and url:  # Only clean up if we downloaded from URL
        print(f"\nCleaning up downloaded slides...")
        try:
            shutil.rmtree(str(downloads_dir))
            print(f"Removed: {downloads_dir}")
        except Exception as e:
            print(f"Warning: Could not remove downloads folder: {e}")
    else:
        if url:
            print(f"\nDownloaded slides kept at: {downloads_dir}")

    return {
        'success': total_success,
        'errors': total_errors,
        'total': len(prompts_data),
        'output_dir': str(generated_dir),
        'time': elapsed_time
    }


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Recreate images from TikTok slideshow using Gemini Vision and Batch API'
    )
    parser.add_argument(
        '--url',
        type=str,
        default='',
        help='TikTok slideshow URL'
    )
    parser.add_argument(
        '--url-file',
        type=str,
        default='',
        help='File containing TikTok URLs (line by line, comma, tab, or space separated)'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='',
        help='Local folder containing pre-downloaded slide images (alternative to --url)'
    )
    parser.add_argument(
        '--skip',
        type=str,
        default='',
        help='Slide number(s) to skip (e.g., "5" or "3,5,7")'
    )
    parser.add_argument(
        '--bulk',
        type=int,
        default=BATCH_SIZE,
        help=f'Number of images per batch (default: {BATCH_SIZE}, max: 50)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='',
        help='Custom output directory name'
    )
    parser.add_argument(
        '--keep',
        action='store_true',
        help='Keep downloaded TikTok images for reference'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all slides without generating'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Download and generate prompts only, do not create images'
    )
    parser.add_argument(
        '--variations',
        type=int,
        default=1,
        help='Number of variations to generate per slide (default: 1)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=3,
        help='Number of parallel workers for processing multiple URLs (default: 3)'
    )

    args = parser.parse_args()

    # Validate that at least one source is provided
    if not args.url and not args.url_file and not args.input_dir:
        print("ERROR: Either --url, --url-file, or --input-dir must be provided!")
        print("\nUsage examples:")
        print("  Single TikTok URL:")
        print("    python tt_recreation.py --url 'https://www.tiktok.com/@user/photo/123456'")
        print("\n  Multiple URLs from file:")
        print("    python tt_recreation.py --url-file urls.txt")
        print("\n  From pre-downloaded images:")
        print("    python tt_recreation.py --input-dir './my_downloaded_slides'")
        print("\nURL file format (any of these work):")
        print("  - One URL per line")
        print("  - Comma separated: url1,url2,url3")
        print("  - Tab or space separated")
        sys.exit(1)

    # Validate batch size
    if args.bulk > 50:
        print("WARNING: Gemini batch API has a limit of 50 requests per batch.")
        print(f"Setting batch size to 50.")
        args.bulk = 50

    # Parse skip list
    skip_list = parse_skip_arg(args.skip)
    if skip_list:
        print(f"Slides to skip: {skip_list}")

    # Create timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Handle URL file (multiple URLs)
    if args.url_file:
        urls = parse_urls_from_file(args.url_file)
        if not urls:
            print("ERROR: No valid TikTok URLs found in file!")
            sys.exit(1)

        print(f"\n{'='*60}")
        print(f"BATCH URL PROCESSING")
        print(f"{'='*60}")
        print(f"Found {len(urls)} TikTok URLs to process")
        print(f"Workers: {args.workers} (parallel processing)")
        for i, url in enumerate(urls, 1):
            print(f"  {i}. {url}")
        print(f"{'='*60}\n")

        # Process URLs in parallel using ThreadPoolExecutor
        all_results = [None] * len(urls)  # Pre-allocate to preserve order
        total_start_time = time.time()

        def process_url_task(idx, url):
            """Wrapper for parallel processing"""
            result = process_source(
                url=url,
                input_dir='',
                skip_list=skip_list,
                bulk=args.bulk,
                output_base=args.output,
                keep=args.keep,
                variations=args.variations,
                list_only=args.list,
                dry_run=args.dry_run,
                timestamp=timestamp,
                url_index=idx,
                total_urls=len(urls)
            )
            return idx, result

        # Determine number of workers (don't exceed number of URLs)
        num_workers = min(args.workers, len(urls))

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_url = {
                executor.submit(process_url_task, idx, url): (idx, url)
                for idx, url in enumerate(urls, 1)
            }

            # Collect results as they complete
            for future in as_completed(future_to_url):
                idx, url = future_to_url[future]
                try:
                    result_idx, result = future.result()
                    all_results[result_idx - 1] = result
                    print(f"\n[Worker] Completed URL {result_idx}/{len(urls)}")
                except Exception as e:
                    print(f"\n[Worker] ERROR processing URL {idx}: {e}")
                    all_results[idx - 1] = {
                        'success': 0,
                        'errors': 1,
                        'total': 0,
                        'error': str(e)
                    }

        # Print final summary
        total_elapsed = time.time() - total_start_time
        total_success = sum(r['success'] for r in all_results if r)
        total_errors = sum(r['errors'] for r in all_results if r)
        total_images = sum(r['total'] for r in all_results if r)

        print(f"\n{'#'*60}")
        print(f"# ALL URLs PROCESSED!")
        print(f"{'#'*60}")
        print(f"Total URLs: {len(urls)}")
        print(f"Parallel workers: {num_workers}")
        print(f"Total images: {total_images}")
        print(f"Total generated: {total_success}")
        print(f"Total errors: {total_errors}")
        print(f"Total time: {total_elapsed:.1f} seconds")
        print(f"{'#'*60}")

        return

    # Single URL or input-dir processing
    result = process_source(
        url=args.url,
        input_dir=args.input_dir,
        skip_list=skip_list,
        bulk=args.bulk,
        output_base=args.output,
        keep=args.keep,
        variations=args.variations,
        list_only=args.list,
        dry_run=args.dry_run,
        timestamp=timestamp
    )


if __name__ == "__main__":
    main()
