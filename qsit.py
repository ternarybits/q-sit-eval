import argparse
from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    LlavaOnevisionForConditionalGeneration,
)


def wa5(logits: dict[str, int]) -> float:
    """
    Convert logits to a weighted score.
    Args:
        logits: The logits to convert.
    Returns:
        The weighted score.
    """
    logprobs = np.array(
        [
            logits["Excellent"],
            logits["Good"],
            logits["Fair"],
            logits["Poor"],
            logits["Bad"],
        ]
    )
    print("logprobs", logprobs)
    probs = np.exp(logprobs) / np.sum(np.exp(logprobs))
    print("probs", probs)
    return np.inner(probs, np.array([1, 0.75, 0.5, 0.25, 0]))


def process_image(
    image_path: Path,
    model: LlavaOnevisionForConditionalGeneration,
    processor: AutoProcessor,
    tokenizer: AutoTokenizer,
    rating_token_dict: dict[str, int],
) -> tuple[Path, float]:
    """
    Process an image using the Q-SiT model and return the score.
    Args:
        image_path: The path to the image to process.
        model: The Q-SiT model to use.
        processor: The processor to use.
        tokenizer: The tokenizer to use.
        rating_token_dict: The dictionary of rating tokens to their IDs.
    Returns:
        The path to the image and the score.
    """
    try:
        # Load image from path
        raw_image = Image.open(image_path)

        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Assume you are an image quality evaluator. \nYour rating should be chosen from the following five categories: Excellent, Good, Fair, Poor, and Bad (from high to low). \nHow would you rate the quality of this image?",
                    },
                    {"type": "image"},
                ],
            },
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

        inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to(
            0, torch.float16
        )

        # Manually append the assistant prefix "The quality of this image is "
        prefix_text = "The quality of this image is "
        prefix_ids = tokenizer(prefix_text, return_tensors="pt")["input_ids"].to(0)
        inputs["input_ids"] = torch.cat([inputs["input_ids"], prefix_ids], dim=-1)
        inputs["attention_mask"] = torch.ones_like(
            inputs["input_ids"]
        )  # Update attention mask

        # Generate exactly one token (the rating)
        output = model.generate(
            **inputs,
            max_new_tokens=1,  # Generate only the rating token
            output_logits=True,
            return_dict_in_generate=True,
        )

        # Extract logits for the generated rating token
        last_logits = output.logits[-1][0]  # Shape: [vocab_size]
        logits_dict = {
            rating: last_logits[rating_token_id].item()
            for rating, rating_token_id in rating_token_dict.items()
        }
        weighted_score = wa5(logits_dict)

        return image_path, weighted_score
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return image_path, None


def get_image_paths(input_path: Path) -> list[Path]:
    """Takes a file or directory and returns all of the image paths in it."""
    image_paths = []
    if input_path.is_file():
        # Process a single file
        image_paths = [input_path]
    elif input_path.is_dir():
        # Process all image files in the directory
        valid_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"]
        image_paths = [
            f for ext in valid_extensions for f in input_path.glob(f"*{ext}")
        ]
    return image_paths


def generate_rating_token_dict(tokenizer: AutoTokenizer) -> dict[str, int]:
    """Generate a dictionary of rating tokens and their IDs."""
    ratings = ["Excellent", "Good", "Fair", "Poor", "Bad"]
    return {
        rating: id[0] for rating, id in zip(ratings, tokenizer(ratings)["input_ids"])
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate image quality using Q-SiT model"
    )
    parser.add_argument(
        "input_path", help="Path to an image file or directory containing images"
    )
    parser.add_argument(
        "--model",
        choices=["q-sit", "q-sit-mini"],
        default="q-sit-mini",
        help="Model version to use (default: q-sit-mini)",
    )
    args = parser.parse_args()

    # Check if input path exists
    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"Error: {input_path} does not exist")
        return

    image_paths = get_image_paths(input_path)
    if not image_paths:
        print(f"No image files found in {input_path}")
        return

    # Load model and tokenizer
    model_id = f"zhangzicheng/{args.model}"
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(0)

    processor = AutoProcessor.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    rating_token_dict = generate_rating_token_dict(tokenizer)
    print("Rating token IDs:", rating_token_dict)

    print(f"Processing {len(image_paths)} image(s)...")
    results = []
    for image_path in image_paths:
        start_time = time.time()
        path, score = process_image(
            image_path, model, processor, tokenizer, rating_token_dict
        )
        elapsed_time = time.time() - start_time
        results.append((path, score, elapsed_time))
        print(
            f"{path.name}: Score (0-1): {score:.4f} (Processed in {elapsed_time:.2f} seconds)"
        )

    # Print summary results if multiple files were processed
    if len(results) > 1:
        avg_score = sum(r[1] for r in results) / len(results)
        avg_time = sum(r[2] for r in results) / len(results)
        total_time = sum(r[2] for r in results)
        print(f"\nAverage score (0-1) across {len(results)} images: {avg_score:.4f}")
        print(f"Average time per image: {avg_time:.2f} seconds")
        print(f"Total time: {total_time:.2f} seconds")


if __name__ == "__main__":
    main()
