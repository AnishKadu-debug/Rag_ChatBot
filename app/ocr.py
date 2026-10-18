"""Optical Character Recognition (OCR) module for scanned document processing.

Why OCR in a Land Records RAG System?
Many Maharashtra land records (like historic 7/12 extracts, Ferfar mutation registers,
or scanned e-Mojni maps) are physical paper scans or photographs converted to PDF.
These documents lack digital text streams: PyMuPDF returns 0 characters for them.

This module provides:
1. Tesseract auto-discovery across standard Windows paths and system PATH.
2. Language pack verification for Marathi ('mar'), English ('eng'), and Hindi ('hin').
3. High-resolution raster image text extraction via pytesseract.
4. Clear error diagnostics when system dependencies are missing.
"""

import logging
import os
from pathlib import Path
import shutil
import subprocess
from typing import List, Optional, Tuple, Union
from PIL import Image
import pytesseract

from app.config import get_settings

logger = logging.getLogger(__name__)

# Standard Windows installation candidates for Tesseract OCR
DEFAULT_WINDOWS_TESSERACT_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
]


class OCRError(Exception):
    """Base exception for all OCR-related operations."""
    pass


class TesseractNotInstalledError(OCRError):
    """Raised when the Tesseract OCR executable cannot be found on the host system."""
    pass


class LanguagePackMissingError(OCRError):
    """Raised when one or more required Tesseract traineddata language packs are missing."""
    pass


def find_tesseract_binary() -> Optional[str]:
    """Locate the Tesseract executable path.

    Checks:
    1. Explicit TESSERACT_CMD configured in settings or environment.
    2. System PATH (via shutil.which).
    3. Standard Windows installation directories.

    Returns:
        Optional[str]: Absolute path to tesseract.exe if found, else None.
    """
    settings = get_settings()

    # 1. Explicit configuration from .env
    if settings.TESSERACT_CMD and Path(settings.TESSERACT_CMD).is_file():
        return settings.TESSERACT_CMD

    # 2. Check system PATH
    which_path = shutil.which("tesseract")
    if which_path:
        return which_path

    # 3. Check known Windows paths
    for path_str in DEFAULT_WINDOWS_TESSERACT_PATHS:
        candidate = Path(path_str)
        if candidate.is_file():
            return str(candidate)

    return None


def is_tesseract_available() -> bool:
    """Check whether Tesseract is installed and runnable on the system.

    Returns:
        bool: True if Tesseract executable exists and responds to --version, False otherwise.
    """
    tess_path = find_tesseract_binary()
    if not tess_path:
        return False

    try:
        result = subprocess.run(
            [tess_path, "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception as exc:
        logger.debug("Failed to execute tesseract at '%s': %s", tess_path, exc)
        return False


def get_available_languages() -> List[str]:
    """Retrieve the list of installed Tesseract language packs (e.g. ['eng', 'mar', 'osd']).

    Returns:
        List[str]: List of lowercase installed language codes.
    """
    tess_path = find_tesseract_binary()
    if not tess_path:
        return []

    try:
        pytesseract.pytesseract.tesseract_cmd = tess_path
        langs = pytesseract.get_languages(config="")
        return [l.strip().lower() for l in langs if l.strip()]
    except Exception as exc:
        logger.warning("Could not list Tesseract languages: %s", exc)
        return []


def check_language_support(target_langs: Union[str, List[str]]) -> Tuple[bool, List[str]]:
    """Check if the requested language codes are installed.

    Args:
        target_langs: Language string (e.g. 'mar+eng' or 'mar') or list of codes.

    Returns:
        Tuple[bool, List[str]]: (all_present, missing_language_codes)
    """
    if isinstance(target_langs, str):
        requested = [l.strip().lower() for l in target_langs.split("+") if l.strip()]
    else:
        requested = [l.strip().lower() for l in target_langs if l.strip()]

    available = get_available_languages()
    missing = [lang for lang in requested if lang not in available]
    return (len(missing) == 0, missing)


def extract_text_from_image(
    image: Image.Image,
    langs: Optional[str] = None,
) -> str:
    """Perform OCR on a PIL image using Tesseract with Devanagari/multilingual support.

    Args:
        image: PIL Image object to perform OCR on.
        langs: Tesseract language string (defaults to config OCR_LANGS, e.g. 'mar+eng').

    Returns:
        str: Extracted text content, stripped of outer whitespace.

    Raises:
        TesseractNotInstalledError: If Tesseract is not installed or not in PATH.
        LanguagePackMissingError: If a requested language pack is missing from tessdata.
        OCRError: If an internal pytesseract execution error occurs.
    """
    settings = get_settings()
    active_langs = langs or settings.OCR_LANGS

    # 1. Locate Tesseract executable
    tess_cmd = find_tesseract_binary()
    if not tess_cmd:
        raise TesseractNotInstalledError(
            "Tesseract OCR executable was not found on your system. "
            "To process scanned documents, please install Tesseract OCR and ensure it is "
            "added to PATH or configured in .env as TESSERACT_CMD."
        )

    pytesseract.pytesseract.tesseract_cmd = tess_cmd

    # 2. Check for missing language packs
    ok, missing = check_language_support(active_langs)
    if not ok:
        raise LanguagePackMissingError(
            f"The required Tesseract language pack(s) {missing} are not installed in tessdata. "
            f"Please download the traineddata file(s) for {missing} and place them in your Tesseract "
            f"tessdata folder (e.g., mar.traineddata for Marathi)."
        )

    # 3. Perform image-to-text conversion
    try:
        # psm 1: Automatic page segmentation with OSD (Orientation and Script Detection)
        # or psm 3: Fully automatic page segmentation (default)
        raw_text = pytesseract.image_to_string(
            image,
            lang=active_langs,
            config="--psm 3",
        )
        return raw_text.strip()
    except Exception as exc:
        logger.error("OCR execution failed: %s", exc)
        raise OCRError(f"Tesseract OCR failed to process image: {exc}") from exc
