"""
I/O utilities for AG News Text Classification Framework.

Provides functions for file operations, model serialization, and data handling.
"""

import os
import json
import yaml
import pickle
import joblib
import shutil
import tarfile
import zipfile
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from contextlib import contextmanager
import tempfile
from urllib.parse import urlparse
import requests
from tqdm import tqdm
import torch
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# File size units
UNITS = {
    "B": 1,
    "KB": 1024,
    "MB": 1024**2,
    "GB": 1024**3,
    "TB": 1024**4,
}

def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def safe_save(
    obj: Any,
    filepath: Union[str, Path],
    backup: bool = True,
    atomic: bool = True
):
    """
    Safely save object to file with optional backup and atomic writing.
    
    Args:
        obj: Object to save
        filepath: File path
        backup: Whether to create backup
        atomic: Whether to use atomic writing
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    
    # Create backup if requested
    if backup and filepath.exists():
        backup_path = filepath.with_suffix(filepath.suffix + ".backup")
        shutil.copy2(filepath, backup_path)
        logger.debug(f"Created backup: {backup_path}")
    
    # Determine save function based on file extension
    ext = filepath.suffix.lower()
    
    try:
        if atomic:
            # Use temporary file for atomic writing
            with tempfile.NamedTemporaryFile(
                mode="wb" if ext in [".pkl", ".joblib", ".pt", ".pth"] else "w",
                dir=filepath.parent,
                delete=False
            ) as tmp_file:
                tmp_path = Path(tmp_file.name)
                
                # Save to temporary file
                _save_object(obj, tmp_path, ext)
                
            # Atomic rename
            tmp_path.replace(filepath)
        else:
            # Direct save
            _save_object(obj, filepath, ext)
        
        logger.info(f"Saved to {filepath}")
        
    except Exception as e:
        logger.error(f"Failed to save {filepath}: {e}")
        # Restore backup if save failed
        if backup and backup_path.exists():
            shutil.copy2(backup_path, filepath)
            logger.info(f"Restored from backup: {backup_path}")
        raise

def _save_object(obj: Any, filepath: Path, ext: str):
    """Internal function to save object based on extension."""
    if ext == ".json":
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
    
    elif ext in [".yaml", ".yml"]:
        with open(filepath, "w", encoding="utf-8") as f:
            yaml.safe_dump(obj, f, default_flow_style=False, allow_unicode=True)
    
    elif ext == ".pkl":
        with open(filepath, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    elif ext == ".joblib":
        joblib.dump(obj, filepath, compress=3)
    
    elif ext in [".pt", ".pth"]:
        torch.save(obj, filepath)
    
    elif ext == ".npy":
        np.save(filepath, obj)
    
    elif ext == ".npz":
        np.savez_compressed(filepath, obj)
    
    elif ext == ".csv":
        if isinstance(obj, pd.DataFrame):
            obj.to_csv(filepath, index=False)
        else:
            pd.DataFrame(obj).to_csv(filepath, index=False)
    
    elif ext == ".parquet":
        if isinstance(obj, pd.DataFrame):
            obj.to_parquet(filepath, engine="pyarrow", compression="snappy")
        else:
            pd.DataFrame(obj).to_parquet(filepath)
    
    elif ext == ".txt":
        with open(filepath, "w", encoding="utf-8") as f:
            if isinstance(obj, (list, tuple)):
                f.write("\n".join(str(item) for item in obj))
            else:
                f.write(str(obj))
    
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

def safe_load(
    filepath: Union[str, Path],
    default: Any = None,
    raise_error: bool = False
) -> Any:
    """
    Safely load object from file.
    
    Args:
        filepath: File path
        default: Default value if file doesn't exist
        raise_error: Whether to raise error on failure
        
    Returns:
        Loaded object or default
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        if raise_error:
            raise FileNotFoundError(f"File not found: {filepath}")
        logger.warning(f"File not found: {filepath}, returning default")
        return default
    
    ext = filepath.suffix.lower()
    
    try:
        return _load_object(filepath, ext)
    except Exception as e:
        logger.error(f"Failed to load {filepath}: {e}")
        if raise_error:
            raise
        return default

def _load_object(filepath: Path, ext: str) -> Any:
    """Internal function to load object based on extension."""
    if ext == ".json":
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    
    elif ext in [".yaml", ".yml"]:
        with open(filepath, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    
    elif ext == ".pkl":
        with open(filepath, "rb") as f:
            return pickle.load(f)
    
    elif ext == ".joblib":
        return joblib.load(filepath)
    
    elif ext in [".pt", ".pth"]:
        return torch.load(filepath, map_location="cpu")
    
    elif ext == ".npy":
        return np.load(filepath, allow_pickle=True)
    
    elif ext == ".npz":
        return np.load(filepath, allow_pickle=True)
    
    elif ext == ".csv":
        return pd.read_csv(filepath)
    
    elif ext == ".parquet":
        return pd.read_parquet(filepath, engine="pyarrow")
    
    elif ext == ".txt":
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

def download_file(
    url: str,
    dest_path: Union[str, Path],
    chunk_size: int = 8192,
    resume: bool = True,
    verify_ssl: bool = True,
    show_progress: bool = True
) -> Path:
    """
    Download file from URL with progress bar and resume support.
    
    Args:
        url: URL to download from
        dest_path: Destination path
        chunk_size: Download chunk size
        resume: Whether to resume partial downloads
        verify_ssl: Whether to verify SSL certificates
        show_progress: Whether to show progress bar
        
    Returns:
        Path to downloaded file
    """
    dest_path = Path(dest_path)
    ensure_dir(dest_path.parent)
    
    # Check if file exists and get size for resume
    resume_pos = 0
    mode = "wb"
    
    if dest_path.exists() and resume:
        resume_pos = dest_path.stat().st_size
        mode = "ab"
    
    # Setup request headers for resume
    headers = {}
    if resume_pos > 0:
        headers["Range"] = f"bytes={resume_pos}-"
    
    # Make request
    response = requests.get(url, headers=headers, stream=True, verify=verify_ssl)
    response.raise_for_status()
    
    # Get total size
    total_size = int(response.headers.get("content-length", 0))
    if resume_pos > 0:
        total_size += resume_pos
    
    # Download with progress
    with open(dest_path, mode) as f:
        if show_progress:
            pbar = tqdm(
                total=total_size,
                initial=resume_pos,
                unit="B",
                unit_scale=True,
                desc=f"Downloading {dest_path.name}"
            )
        
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                if show_progress:
                    pbar.update(len(chunk))
        
        if show_progress:
            pbar.close()
    
    logger.info(f"Downloaded {url} to {dest_path}")
    return dest_path

def extract_archive(
    archive_path: Union[str, Path],
    extract_to: Optional[Union[str, Path]] = None,
    remove_archive: bool = False
) -> Path:
    """
    Extract archive file.
    
    Args:
        archive_path: Path to archive
        extract_to: Extraction directory
        remove_archive: Whether to remove archive after extraction
        
    Returns:
        Path to extracted directory
    """
    archive_path = Path(archive_path)
    
    if extract_to is None:
        extract_to = archive_path.parent / archive_path.stem
    else:
        extract_to = Path(extract_to)
    
    ensure_dir(extract_to)
    
    # Determine archive type and extract
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
    
    elif archive_path.suffix in [".tar", ".gz", ".bz2", ".xz"]:
        with tarfile.open(archive_path, "r:*") as tar_ref:
            tar_ref.extractall(extract_to)
    
    else:
        raise ValueError(f"Unsupported archive format: {archive_path.suffix}")
    
    logger.info(f"Extracted {archive_path} to {extract_to}")
    
    # Remove archive if requested
    if remove_archive:
        archive_path.unlink()
        logger.info(f"Removed archive: {archive_path}")
    
    return extract_to

def compute_file_hash(
    filepath: Union[str, Path],
    algorithm: str = "sha256",
    chunk_size: int = 65536
) -> str:
    """
    Compute hash of file.
    
    Args:
        filepath: File path
        algorithm: Hash algorithm
        chunk_size: Read chunk size
        
    Returns:
        Hash string
    """
    filepath = Path(filepath)
    
    hasher = hashlib.new(algorithm)
    
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hasher.update(chunk)
    
    return hasher.hexdigest()

def get_file_size(filepath: Union[str, Path], unit: str = "MB") -> float:
    """
    Get file size in specified unit.
    
    Args:
        filepath: File path
        unit: Size unit (B, KB, MB, GB, TB)
        
    Returns:
        File size in specified unit
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        return 0.0
    
    size_bytes = filepath.stat().st_size
    return size_bytes / UNITS.get(unit, UNITS["MB"])

@contextmanager
def temp_directory(cleanup: bool = True):
    """
    Context manager for temporary directory.
    
    Args:
        cleanup: Whether to cleanup on exit
        
    Yields:
        Path to temporary directory
    """
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        yield temp_dir
    finally:
        if cleanup and temp_dir.exists():
            shutil.rmtree(temp_dir)

@contextmanager
def atomic_write(filepath: Union[str, Path], mode: str = "w", **kwargs):
    """
    Context manager for atomic file writing.
    
    Args:
        filepath: Target file path
        mode: File mode
        **kwargs: Additional arguments for open()
        
    Yields:
        File handle
    """
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(
        mode=mode,
        dir=filepath.parent,
        delete=False,
        **kwargs
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)
        
        try:
            yield tmp_file
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        except Exception:
            tmp_path.unlink()
            raise
    
    # Atomic rename
    tmp_path.replace(filepath)

def find_files(
    directory: Union[str, Path],
    pattern: str = "*",
    recursive: bool = True
) -> List[Path]:
    """
    Find files matching pattern.
    
    Args:
        directory: Directory to search
        pattern: File pattern
        recursive: Whether to search recursively
        
    Returns:
        List of matching file paths
    """
    directory = Path(directory)
    
    if recursive:
        return sorted(directory.rglob(pattern))
    else:
        return sorted(directory.glob(pattern))

def copy_tree(
    src: Union[str, Path],
    dst: Union[str, Path],
    ignore_patterns: Optional[List[str]] = None
):
    """
    Copy directory tree.
    
    Args:
        src: Source directory
        dst: Destination directory
        ignore_patterns: Patterns to ignore
    """
    src = Path(src)
    dst = Path(dst)
    
    if ignore_patterns:
        ignore = shutil.ignore_patterns(*ignore_patterns)
    else:
        ignore = None
    
    shutil.copytree(src, dst, ignore=ignore, dirs_exist_ok=True)
    logger.info(f"Copied {src} to {dst}")

def cleanup_directory(
    directory: Union[str, Path],
    keep_latest: int = 0,
    pattern: str = "*"
):
    """
    Clean up directory by removing old files.
    
    Args:
        directory: Directory to clean
        keep_latest: Number of latest files to keep
        pattern: File pattern to match
    """
    directory = Path(directory)
    
    files = find_files(directory, pattern, recursive=False)
    
    if keep_latest > 0:
        # Sort by modification time
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        files_to_remove = files[keep_latest:]
    else:
        files_to_remove = files
    
    for filepath in files_to_remove:
        filepath.unlink()
        logger.debug(f"Removed {filepath}")
    
    if files_to_remove:
        logger.info(f"Cleaned up {len(files_to_remove)} files from {directory}")

# Export public API
__all__ = [
    "ensure_dir",
    "safe_save",
    "safe_load",
    "download_file",
    "extract_archive",
    "compute_file_hash",
    "get_file_size",
    "temp_directory",
    "atomic_write",
    "find_files",
    "copy_tree",
    "cleanup_directory",
]
