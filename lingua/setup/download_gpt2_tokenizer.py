import os
import argparse
from typing import Optional

import requests
from requests.exceptions import HTTPError

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    raise ImportError(
        "huggingface_hub not found. Please install via `pip install huggingface_hub`."
    )

# A small mapping to known GPT-2 Hugging Face model IDs / filenames
GPT2_TOKENIZER_REPOS = {
    "gpt2": ("gpt2", "tokenizer.json"),
    "gpt2-medium": ("gpt2-medium", "tokenizer.json"),
    "gpt2-large": ("gpt2-large", "tokenizer.json"),
    "gpt2-xl": ("gpt2-xl", "tokenizer.json"),
}

def main(
    tokenizer_name: str,
    path_to_save: str,
    api_key: Optional[str] = None,
):
    """
    Download GPT-2 tokenizer or one of its variants from the Hugging Face Hub.
    Saves the retrieved tokenizer.json (and merges file) into 'path_to_save'.
    """
    if tokenizer_name not in GPT2_TOKENIZER_REPOS:
        raise ValueError(
            f"Unknown GPT-2 tokenizer '{tokenizer_name}'. Supported: "
            f"{list(GPT2_TOKENIZER_REPOS.keys())}"
        )
    repo_id, filename = GPT2_TOKENIZER_REPOS[tokenizer_name]

    # Create output directory if it doesn’t exist
    os.makedirs(path_to_save, exist_ok=True)

    print(f"Downloading tokenizer '{tokenizer_name}' from HF repo '{repo_id}' to {path_to_save}...")
    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=path_to_save,
            local_dir_use_symlinks=False,
            token=api_key if api_key else None,
        )
        # Optionally, also download merges.txt, vocab.json, etc. if needed:
        merges_file = "merges.txt"
        vocab_file = "vocab.json"
        # Attempt to download these as well
        for f in [merges_file, vocab_file]:
            try:
                hf_hub_download(
                    repo_id=repo_id,
                    filename=f,
                    local_dir=path_to_save,
                    local_dir_use_symlinks=False,
                    token=api_key if api_key else None,
                )
            except HTTPError:
                # Not all GPT-2 repos have merges.txt/vocab.json in the standard location
                pass

        print("Tokenizer download complete.")
    except HTTPError as e:
        if e.response.status_code == 401:
            print(
                "You need to pass a valid `--api_key=...` to download private models/tokenizers."
            )
        elif e.response.status_code == 403:
            print("Access to this tokenizer is restricted.")
        else:
            raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tokenizer_name", type=str, help="Name of GPT-2 tokenizer (e.g. gpt2, gpt2-medium, etc.)")
    parser.add_argument("tokenizer_dir", type=str, help="Directory to save the tokenizer files.")
    parser.add_argument("--api_key", type=str, default="", help="Optional Hugging Face API token.")
    args = parser.parse_args()

    main(
        tokenizer_name=args.tokenizer_name,
        path_to_save=args.tokenizer_dir,
        api_key=args.api_key
    ) 