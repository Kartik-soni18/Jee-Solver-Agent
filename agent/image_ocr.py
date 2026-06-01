
import base64
import io
import logging
from functools import lru_cache

from PIL import Image

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_pix2text_model():
    """Load and cache the Pix2Text model globally.

    Cached so that repeated OCR calls (e.g. across Streamlit reruns)
    do not reload the heavy model weights.
    """
    from pix2text import Pix2Text

    return Pix2Text.from_config()


class MathImageOCR:
    def __init__(self):
        pass

    def extract(self, image_data: str) -> str:
        try:
            img_bytes = base64.b64decode(image_data)
            img = Image.open(io.BytesIO(img_bytes))
            p2t = _load_pix2text_model()
            result = p2t.recognize_text_formula(img, return_text=True)
            text = result.get("text", "") if isinstance(result, dict) else str(result)
            return text.strip()
        except Exception as exc:
            logger.error("OCR extraction failed: %s", exc)
            return ""
