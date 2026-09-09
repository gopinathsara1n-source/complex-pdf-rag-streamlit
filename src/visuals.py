from pathlib import Path
import json
import random
import time
import re

from google import genai
from google.genai import types

from .config import (
    GEMINI_API_KEY_ENV,
    GEMINI_VISUAL_MODELS,
    MAX_RETRIES,
    INITIAL_RETRY_DELAY,
    MAX_RETRY_DELAY,
    DELAY_BETWEEN_VISUAL_REQUESTS,
)


def build_visual_prompt(page_number: int) -> str:
    return f"""
You are analyzing a visual extracted from page {page_number}
of a complex PDF document for a Retrieval-Augmented Generation
(RAG) system.

Create a highly accurate, self-contained description of the visual.

Include, when visible:
1. The chart or figure title.
2. The type of visual.
3. Important categories, labels and legends.
4. Important numerical values.
5. Dates, years or periods.
6. Units.
7. Trends, comparisons and relationships.
8. Forecast values, if present.
9. Important annotations.
10. The source, if visible.
11. The meaning or interpretation directly supported by the visual.

Do NOT invent information that is not visible.
Do NOT provide opinions or recommendations.
Do NOT refer to yourself as an AI.
Return ONLY the description.
""".strip()


def classify_gemini_error(error):
    text = str(error).upper()

    if any(x in text for x in ["401", "403", "UNAUTHENTICATED", "PERMISSION_DENIED", "INVALID API KEY"]):
        return "authentication"
    if any(x in text for x in ["429", "RESOURCE_EXHAUSTED", "RATE LIMIT", "QUOTA", "TOO MANY REQUESTS"]):
        return "quota"
    if any(x in text for x in ["503", "UNAVAILABLE", "SERVICE UNAVAILABLE", "SERVER BUSY", "INTERNAL", "500", "502", "504"]):
        return "temporary"
    return "permanent"


def call_visual_model(client, image_path: Path, page_number: int, model_name: str):
    prompt = build_visual_prompt(page_number)
    image_bytes = image_path.read_bytes()

    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                    prompt,
                ],
            )
            if not response.text:
                raise ValueError("Gemini returned an empty response.")
            return response.text.strip(), "success"

        except Exception as exc:
            error_type = classify_gemini_error(exc)
            if error_type == "authentication":
                return None, error_type
            if error_type == "permanent":
                return None, error_type

            if attempt < MAX_RETRIES - 1:
                delay = min(
                    INITIAL_RETRY_DELAY * (2 ** attempt) + random.uniform(0, 1),
                    MAX_RETRY_DELAY,
                )
                time.sleep(delay)

    return None, error_type


def generate_visual_descriptions(visual_manifest, output_path: Path, api_key: str):
    if not api_key:
        raise ValueError("GEMINI_API_KEY is required to process visual content.")

    client = genai.Client(api_key=api_key)

    existing = {}
    if output_path.exists():
        try:
            existing = {
                x["picture_id"]: x
                for x in json.loads(output_path.read_text(encoding="utf-8"))
            }
        except Exception:
            existing = {}

    results = dict(existing)

    for item in visual_manifest:
        picture_id = item["picture_id"]
        if results.get(picture_id, {}).get("description"):
            continue

        image_path = Path(item["image_path"])
        if not image_path.exists():
            results[picture_id] = {
                **item,
                "description": None,
                "model": None,
                "status": "image_missing",
            }
            continue

        final = None
        for model_name in GEMINI_VISUAL_MODELS:
            description, status = call_visual_model(
                client, image_path, item["page"], model_name
            )
            if description:
                final = {
                    **item,
                    "model": model_name,
                    "description": description,
                    "status": "success",
                }
                break
            if status == "authentication":
                final = {
                    **item,
                    "model": model_name,
                    "description": None,
                    "status": "authentication_error",
                }
                break

        if final is None:
            final = {
                **item,
                "model": None,
                "description": None,
                "status": "failed",
            }

        results[picture_id] = final
        output_path.write_text(
            json.dumps(
                sorted(results.values(), key=lambda x: (x.get("page", 0), x.get("picture_id", ""))),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        time.sleep(DELAY_BETWEEN_VISUAL_REQUESTS)

    return list(results.values())
