# Cycle 37: File Uploads & Media Processing (January 2026)

## Overview
Comprehensive patterns for handling file uploads and media processing in Python production systems. Covers FastAPI streaming uploads, S3 direct uploads with presigned URLs, and image/video processing pipelines.

---

## 1. FastAPI File Uploads

### Basic UploadFile Pattern
```python
from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
import aiofiles

app = FastAPI()

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    # Validate file type
    allowed_types = {"image/jpeg", "image/png", "application/pdf"}
    if file.content_type not in allowed_types:
        raise HTTPException(400, f"File type {file.content_type} not allowed")
    
    # Validate file size (read in chunks)
    max_size = 10 * 1024 * 1024  # 10MB
    size = 0
    chunks = []
    
    while chunk := await file.read(1024 * 1024):  # 1MB chunks
        size += len(chunk)
        if size > max_size:
            raise HTTPException(413, "File too large")
        chunks.append(chunk)
    
    # Reset file position for reprocessing
    await file.seek(0)
    
    # Save file
    async with aiofiles.open(f"uploads/{file.filename}", "wb") as f:
        for chunk in chunks:
            await f.write(chunk)
    
    return {"filename": file.filename, "size": size}
```

### Streaming Large Files (Memory Efficient)
```python
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
import aiofiles
import hashlib

@app.post("/upload/stream/")
async def stream_upload(request: Request):
    """Handle large uploads without loading into memory"""
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > 5 * 1024 * 1024 * 1024:  # 5GB
        raise HTTPException(413, "File exceeds 5GB limit")
    
    filename = request.headers.get("x-filename", "upload.bin")
    hasher = hashlib.sha256()
    total_size = 0
    
    async with aiofiles.open(f"uploads/{filename}", "wb") as f:
        async for chunk in request.stream():
            await f.write(chunk)
            hasher.update(chunk)
            total_size += len(chunk)
    
    return {
        "filename": filename,
        "size": total_size,
        "sha256": hasher.hexdigest()
    }
```

### Multiple File Upload
```python
@app.post("/upload/batch/")
async def upload_multiple(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        content = await file.read()
        results.append({
            "filename": file.filename,
            "size": len(content),
            "type": file.content_type
        })
    return {"files": results, "count": len(results)}
```

---

## 2. S3 Direct Uploads with Presigned URLs

### Generate Presigned URL for Client Upload
```python
import boto3
from botocore.config import Config
from datetime import datetime
import uuid

s3_client = boto3.client(
    "s3",
    config=Config(signature_version="s3v4"),
    region_name="us-east-1"
)

async def generate_upload_url(
    filename: str,
    content_type: str,
    max_size_mb: int = 100
) -> dict:
    """Generate presigned POST for direct browser upload"""
    
    # Generate unique key
    ext = filename.rsplit(".", 1)[-1] if "." in filename else ""
    key = f"uploads/{datetime.now():%Y/%m/%d}/{uuid.uuid4()}.{ext}"
    
    # Presigned POST with conditions
    presigned = s3_client.generate_presigned_post(
        Bucket="my-bucket",
        Key=key,
        Fields={
            "Content-Type": content_type,
            "x-amz-meta-original-name": filename
        },
        Conditions=[
            {"Content-Type": content_type},
            ["content-length-range", 1, max_size_mb * 1024 * 1024],
            {"x-amz-meta-original-name": filename}
        ],
        ExpiresIn=3600  # 1 hour
    )
    
    return {
        "upload_url": presigned["url"],
        "fields": presigned["fields"],
        "key": key,
        "expires_in": 3600
    }
```

### Multipart Upload for Large Files (20GB+)
```python
from boto3.s3.transfer import TransferConfig

# Configure multipart thresholds
transfer_config = TransferConfig(
    multipart_threshold=100 * 1024 * 1024,  # 100MB
    max_concurrency=10,
    multipart_chunksize=100 * 1024 * 1024,  # 100MB chunks
    use_threads=True
)

async def initiate_multipart_upload(filename: str) -> dict:
    """Start multipart upload and return upload_id"""
    key = f"large-uploads/{uuid.uuid4()}/{filename}"
    
    response = s3_client.create_multipart_upload(
        Bucket="my-bucket",
        Key=key,
        ContentType="application/octet-stream"
    )
    
    return {
        "upload_id": response["UploadId"],
        "key": key
    }

async def generate_part_url(key: str, upload_id: str, part_number: int) -> str:
    """Generate presigned URL for uploading a single part"""
    return s3_client.generate_presigned_url(
        "upload_part",
        Params={
            "Bucket": "my-bucket",
            "Key": key,
            "UploadId": upload_id,
            "PartNumber": part_number
        },
        ExpiresIn=3600
    )

async def complete_multipart_upload(
    key: str,
    upload_id: str,
    parts: list[dict]  # [{"PartNumber": 1, "ETag": "..."}, ...]
) -> dict:
    """Complete the multipart upload"""
    response = s3_client.complete_multipart_upload(
        Bucket="my-bucket",
        Key=key,
        UploadId=upload_id,
        MultipartUpload={"Parts": parts}
    )
    return {"location": response["Location"], "key": key}
```

### Frontend Multipart Upload Flow
```typescript
// Client-side multipart upload
async function uploadLargeFile(file: File) {
    const CHUNK_SIZE = 100 * 1024 * 1024; // 100MB
    const totalParts = Math.ceil(file.size / CHUNK_SIZE);
    
    // 1. Initiate upload
    const { upload_id, key } = await api.post("/upload/multipart/init", {
        filename: file.name
    });
    
    // 2. Upload parts in parallel (limited concurrency)
    const parts = [];
    for (let i = 0; i < totalParts; i++) {
        const start = i * CHUNK_SIZE;
        const end = Math.min(start + CHUNK_SIZE, file.size);
        const chunk = file.slice(start, end);
        
        const { url } = await api.post("/upload/multipart/part-url", {
            key, upload_id, part_number: i + 1
        });
        
        const response = await fetch(url, {
            method: "PUT",
            body: chunk
        });
        
        parts.push({
            PartNumber: i + 1,
            ETag: response.headers.get("ETag")
        });
    }
    
    // 3. Complete upload
    return api.post("/upload/multipart/complete", {
        key, upload_id, parts
    });
}
```

---

## 3. Image Processing with Pillow

### Resize and Thumbnail Generation
```python
from PIL import Image
from io import BytesIO
import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=4)

def process_image_sync(
    image_bytes: bytes,
    max_width: int = 1920,
    max_height: int = 1080,
    quality: int = 85
) -> tuple[bytes, bytes]:
    """Process image: resize + generate thumbnail"""
    
    img = Image.open(BytesIO(image_bytes))
    
    # Convert to RGB if necessary (handles PNG with alpha)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    
    # Resize maintaining aspect ratio
    img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
    
    # Save resized
    resized_buffer = BytesIO()
    img.save(resized_buffer, format="JPEG", quality=quality, optimize=True)
    resized_bytes = resized_buffer.getvalue()
    
    # Generate thumbnail
    thumb = img.copy()
    thumb.thumbnail((300, 300), Image.Resampling.LANCZOS)
    thumb_buffer = BytesIO()
    thumb.save(thumb_buffer, format="JPEG", quality=80)
    thumb_bytes = thumb_buffer.getvalue()
    
    return resized_bytes, thumb_bytes

async def process_image(image_bytes: bytes) -> tuple[bytes, bytes]:
    """Async wrapper for image processing"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor,
        process_image_sync,
        image_bytes
    )
```

### Image Validation and Metadata
```python
from PIL import Image, ExifTags
from io import BytesIO

def validate_and_extract_metadata(image_bytes: bytes) -> dict:
    """Validate image and extract EXIF metadata"""
    
    try:
        img = Image.open(BytesIO(image_bytes))
        img.verify()  # Verify it's a valid image
        
        # Reopen after verify (verify closes file)
        img = Image.open(BytesIO(image_bytes))
    except Exception as e:
        raise ValueError(f"Invalid image: {e}")
    
    # Extract EXIF
    exif_data = {}
    if hasattr(img, "_getexif") and img._getexif():
        for tag_id, value in img._getexif().items():
            tag = ExifTags.TAGS.get(tag_id, tag_id)
            exif_data[tag] = str(value)
    
    return {
        "format": img.format,
        "mode": img.mode,
        "width": img.width,
        "height": img.height,
        "exif": exif_data
    }
```

### WebP Conversion for Web Optimization
```python
def convert_to_webp(image_bytes: bytes, quality: int = 80) -> bytes:
    """Convert image to WebP format for smaller file sizes"""
    
    img = Image.open(BytesIO(image_bytes))
    
    if img.mode in ("RGBA", "P"):
        # Preserve transparency for WebP
        pass
    else:
        img = img.convert("RGB")
    
    buffer = BytesIO()
    img.save(
        buffer,
        format="WEBP",
        quality=quality,
        method=6  # Best compression
    )
    
    return buffer.getvalue()
```

---

## 4. Video Processing Pipeline

### FFmpeg Integration for Transcoding
```python
import subprocess
import tempfile
import os
from pathlib import Path

async def transcode_video(
    input_path: str,
    output_format: str = "mp4",
    resolution: str = "1920x1080",
    bitrate: str = "5M"
) -> str:
    """Transcode video using FFmpeg"""
    
    output_path = f"{input_path.rsplit('.', 1)[0]}_transcoded.{output_format}"
    
    cmd = [
        "ffmpeg", "-i", input_path,
        "-vf", f"scale={resolution}:force_original_aspect_ratio=decrease",
        "-c:v", "libx264", "-preset", "medium",
        "-b:v", bitrate,
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",  # Web optimization
        "-y", output_path
    ]
    
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    stdout, stderr = await process.communicate()
    
    if process.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {stderr.decode()}")
    
    return output_path

async def generate_video_thumbnail(
    video_path: str,
    timestamp: str = "00:00:01"
) -> bytes:
    """Extract thumbnail frame from video"""
    
    cmd = [
        "ffmpeg", "-i", video_path,
        "-ss", timestamp,
        "-vframes", "1",
        "-f", "image2pipe",
        "-vcodec", "mjpeg",
        "-"
    ]
    
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    stdout, stderr = await process.communicate()
    
    if process.returncode != 0:
        raise RuntimeError(f"Thumbnail extraction failed: {stderr.decode()}")
    
    return stdout
```

---

## 5. File Validation & Security

### Comprehensive File Validator
```python
import magic
import hashlib
from typing import Optional

ALLOWED_TYPES = {
    "image/jpeg": {"extensions": [".jpg", ".jpeg"], "max_size": 20 * 1024 * 1024},
    "image/png": {"extensions": [".png"], "max_size": 20 * 1024 * 1024},
    "image/webp": {"extensions": [".webp"], "max_size": 20 * 1024 * 1024},
    "application/pdf": {"extensions": [".pdf"], "max_size": 50 * 1024 * 1024},
    "video/mp4": {"extensions": [".mp4"], "max_size": 500 * 1024 * 1024},
}

class FileValidator:
    def __init__(self, allowed_types: dict = ALLOWED_TYPES):
        self.allowed_types = allowed_types
        self.magic = magic.Magic(mime=True)
    
    def validate(
        self,
        file_bytes: bytes,
        filename: str,
        claimed_type: Optional[str] = None
    ) -> dict:
        """Validate file thoroughly"""
        
        # 1. Detect actual MIME type from content
        actual_type = self.magic.from_buffer(file_bytes)
        
        # 2. Check if type is allowed
        if actual_type not in self.allowed_types:
            raise ValueError(f"File type {actual_type} not allowed")
        
        # 3. Verify claimed type matches actual type
        if claimed_type and claimed_type != actual_type:
            raise ValueError(f"Type mismatch: claimed {claimed_type}, actual {actual_type}")
        
        # 4. Check extension
        config = self.allowed_types[actual_type]
        ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext and ext not in config["extensions"]:
            raise ValueError(f"Extension {ext} not valid for {actual_type}")
        
        # 5. Check size
        size = len(file_bytes)
        if size > config["max_size"]:
            raise ValueError(f"File size {size} exceeds limit {config['max_size']}")
        
        return {
            "mime_type": actual_type,
            "size": size,
            "hash": hashlib.sha256(file_bytes).hexdigest()
        }
```

### Filename Sanitization
```python
import re
import unicodedata
from pathlib import Path

def sanitize_filename(filename: str, max_length: int = 200) -> str:
    """Sanitize filename for safe storage"""
    
    # Normalize unicode
    filename = unicodedata.normalize("NFKD", filename)
    
    # Remove non-ASCII characters
    filename = filename.encode("ascii", "ignore").decode("ascii")
    
    # Replace spaces and dangerous characters
    filename = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", filename)
    
    # Remove leading/trailing dots and spaces
    filename = filename.strip(". ")
    
    # Limit length (preserve extension)
    if len(filename) > max_length:
        name, ext = filename.rsplit(".", 1) if "." in filename else (filename, "")
        max_name = max_length - len(ext) - 1
        filename = f"{name[:max_name]}.{ext}" if ext else name[:max_length]
    
    # Fallback if empty
    return filename or "unnamed_file"
```

---

## 6. Production Patterns

### Complete Upload Endpoint with Processing Pipeline
```python
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from pydantic import BaseModel
import uuid

class UploadResponse(BaseModel):
    id: str
    filename: str
    size: int
    status: str
    urls: dict

@app.post("/api/upload/", response_model=UploadResponse)
async def upload_with_processing(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    # Read file
    content = await file.read()
    
    # Validate
    validator = FileValidator()
    try:
        validation = validator.validate(
            content,
            file.filename,
            file.content_type
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    
    # Generate unique ID
    upload_id = str(uuid.uuid4())
    
    # Store original to S3
    key = f"uploads/{upload_id}/original"
    s3_client.put_object(
        Bucket="my-bucket",
        Key=key,
        Body=content,
        ContentType=validation["mime_type"]
    )
    
    # Queue background processing
    if validation["mime_type"].startswith("image/"):
        background_tasks.add_task(
            process_image_background,
            upload_id,
            content
        )
    
    return UploadResponse(
        id=upload_id,
        filename=sanitize_filename(file.filename),
        size=validation["size"],
        status="processing",
        urls={
            "original": f"/api/files/{upload_id}/original"
        }
    )

async def process_image_background(upload_id: str, content: bytes):
    """Background task for image processing"""
    resized, thumbnail = await process_image(content)
    
    # Store processed versions
    s3_client.put_object(
        Bucket="my-bucket",
        Key=f"uploads/{upload_id}/resized",
        Body=resized,
        ContentType="image/jpeg"
    )
    s3_client.put_object(
        Bucket="my-bucket",
        Key=f"uploads/{upload_id}/thumbnail",
        Body=thumbnail,
        ContentType="image/jpeg"
    )
    
    # Update status in database
    await db.update_upload_status(upload_id, "completed")
```

---

## 7. Decision Matrix

| Scenario | Approach |
|----------|----------|
| Small files (<10MB) | FastAPI UploadFile, in-memory |
| Medium files (10-100MB) | Streaming upload, disk buffer |
| Large files (100MB-5GB) | S3 presigned URL, direct upload |
| Very large (5GB+) | S3 multipart, chunked upload |
| Image optimization | Pillow + WebP conversion |
| Video transcoding | FFmpeg subprocess |
| Background processing | FastAPI BackgroundTasks or Celery |

---

## 8. Anti-Patterns to Avoid

1. **Loading full file into memory** for large uploads
2. **Trusting Content-Type header** - always detect from content
3. **Using original filename** directly - always sanitize
4. **Missing file size limits** - enables DoS attacks
5. **Synchronous video processing** - blocks event loop
6. **No virus scanning** for production systems
7. **Missing cleanup** for failed partial uploads
8. **Presigned URLs without expiry** - security risk
9. **No idempotency** for multipart uploads - leads to duplicates
10. **Storing files in DB** instead of object storage

---

## 9. Production Checklist

- [ ] File type validation (magic bytes, not extension)
- [ ] File size limits per type
- [ ] Filename sanitization
- [ ] Virus/malware scanning (ClamAV or cloud)
- [ ] S3 presigned URLs for large files
- [ ] Background processing for media
- [ ] CDN for serving files
- [ ] Cleanup job for orphaned files
- [ ] Rate limiting on upload endpoints
- [ ] Logging and monitoring
- [ ] Backup strategy for uploads

---

*Cycle 37 - File Uploads & Media Processing | January 2026*
*Research synthesized from FastAPI, boto3, Pillow, FFmpeg patterns*
