# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
import os
import time
import subprocess
import requests
from huggingface_hub import snapshot_download
import glob
import random
import multiprocessing
from functools import partial

def run_command(command):
    print(f"Running: {command}")
    subprocess.run(command, shell=True, check=True)


def download_file(url, local_dir):
    try:
        url = url.strip()
        if not url:
            return
            
        # Extract the relative path from the URL
        rel_path = url.replace('https://data.together.xyz/redpajama-data-1T/v1.0.0/', '')
        output_path = os.path.join(local_dir, rel_path)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Download file with wget
        if not os.path.exists(output_path):
            print(f"Downloading: {url}")
            run_command(f"wget -q --show-progress {url} -O {output_path}")
        else:
            print(f"File already exists, skipping: {output_path}")
            
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        return url
    return None


def download_dataset(repo_id, local_dir, allow_patterns):
    print(f"Downloading dataset from {repo_id}...")
    print(f"Looking for files matching patterns: {allow_patterns}")
    max_retries = 5
    retry_delay = 10  # seconds
    
    token = os.getenv("HF_TOKEN")
    if token is None:
        print("Warning: HF_TOKEN not found in environment. Some datasets might be inaccessible.")
        print("Please run: export HF_TOKEN=your_huggingface_token")
    else:
        print("HF_TOKEN found in environment")
    
    for attempt in range(max_retries):
        try:
            # Add verbose logging
            result = snapshot_download(
                repo_id,
                repo_type="dataset",
                local_dir=local_dir,
                allow_patterns=allow_patterns,
                token=token,
                resume_download=True,
                max_workers=4,
                tqdm_class=None,
                revision="main",
            )
            print(f"Download result path: {result}")
            break
        except Exception as e:
            print(f"Detailed error: {str(e)}")
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} of {max_retries}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                raise Exception(f"Failed to download after {max_retries} attempts")
    print(f"Dataset downloaded to {local_dir}")


def parquet_to_jsonl(dataset, work_dir, src_dir, tgt_dir, content_key="text", ntasks=64):
    from datatrove.executor import LocalPipelineExecutor
    from datatrove.pipeline.readers import ParquetReader
    from datatrove.pipeline.writers import JsonlWriter

    pipeline_exec = LocalPipelineExecutor(
        pipeline=[
            ParquetReader(
                src_dir,
                file_progress=True,
                doc_progress=True,
                glob_pattern="**/*.parquet",
                text_key=content_key,
            ),
            JsonlWriter(
                tgt_dir,
                output_filename=dataset + ".chunk.${rank}.jsonl",
                compression=None,
            ),
        ],
        tasks=ntasks,
        logging_dir=os.path.join(work_dir, "datatrove"),
    )
    pipeline_exec.run()


def setup_terashuf(work_dir):
    terashuf_dir = os.path.join(work_dir, "terashuf")
    terashuf_executable = os.path.join(terashuf_dir, "terashuf")

    if os.path.exists(terashuf_executable):
        print("terashuf executable already exists. Skipping setup.")
        return terashuf_dir

    print("Setting up terashuf...")
    run_command(f"git clone https://github.com/alexandres/terashuf {terashuf_dir}")
    run_command(f"make -C {terashuf_dir}")
    return terashuf_dir


def download_redpajama(local_dir, num_processes=32):
    print("Downloading RedPajama dataset using direct download method...")
    urls_file = os.path.join(local_dir, "urls.txt")
    failed_urls_file = os.path.join(local_dir, "failed_urls.txt")
    
    # Download the urls file
    urls_url = 'https://data.together.xyz/redpajama-data-1T/v1.0.0/urls.txt'
    print(f"Downloading urls file from: {urls_url}")
    
    try:
        response = requests.get(urls_url)
        response.raise_for_status()
        
        # Save urls file
        os.makedirs(local_dir, exist_ok=True)
        with open(urls_file, 'w') as f:
            f.write(response.text)
        
        # Read URLs
        with open(urls_file, 'r') as f:
            urls = [url.strip() for url in f.readlines() if url.strip()]
        
        print(f"Starting parallel download with {num_processes} processes...")
        
        # Create a pool of workers
        with multiprocessing.Pool(processes=num_processes) as pool:
            # Use partial to fix the local_dir parameter
            download_func = partial(download_file, local_dir=local_dir)
            
            # Map the download function to URLs and collect failed URLs
            failed_urls = list(filter(None, pool.map(download_func, urls)))
            
        # Report and save any failures
        if failed_urls:
            print(f"\nFailed to download {len(failed_urls)} files. Saving to {failed_urls_file}")
            with open(failed_urls_file, 'w') as f:
                for url in failed_urls:
                    f.write(f"{url}\n")
            print(f"To retry failed downloads later, run: while read url; do wget $url; done < {failed_urls_file}")
            raise Exception(f"Failed to download {len(failed_urls)} files. See {failed_urls_file} for details.")
        
        print("All downloads completed successfully!")
                
    except Exception as e:
        raise Exception(f"Failed to download RedPajama dataset: {str(e)}")


def main(dataset, memory, data_dir, seed=42, nchunks=32, batch_size=5, num_processes=32):
    # Configuration
    repo_id = {
        "fineweb_edu": "HuggingFaceFW/fineweb-edu",
        "fineweb_edu_10bt": "HuggingFaceFW/fineweb-edu",
        "fineweb_edu_100bt": "HuggingFaceFW/fineweb-edu",
        "dclm_baseline_1.0": "mlfoundations/dclm-baseline-1.0",
        "dclm_baseline_1.0_10prct": "mlfoundations/dclm-baseline-1.0",
        "pile": "EleutherAI/pile",
        "pile_deduplicated": "EleutherAI/the_pile_deduplicated",
        "pile_uc": "monology/pile-uncopyrighted",
        "c4": "allenai/c4",
        "redpajama": "togethercomputer/RedPajama-Data-1T",
        "refineweb": "tiiuae/falcon-refinedweb",
        "slimpajama": "cerebras/SlimPajama-627B"
    }[dataset]

    # Use temporary directory for initial processing
    tmp_src_dir = f"/tmp/{dataset}"
    tmp_out_dir = f"/tmp/{dataset}_shuffled"
    
    # Final destination directories
    final_src_dir = f"{data_dir}/{dataset}"
    final_out_dir = f"{data_dir}/{dataset}_shuffled"
    
    os.makedirs(tmp_src_dir, exist_ok=True)
    os.makedirs(tmp_out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(final_src_dir), exist_ok=True)

    # Rest of the configurations remain exactly the same
    work_dir = tmp_src_dir
    prefix = f"{dataset}.chunk."

    orig_extension = {
        "fineweb_edu": ".jsonl",
        "fineweb_edu_10bt": ".jsonl",
        "fineweb_edu_100bt": ".jsonl",
        "dclm_baseline_1.0": ".jsonl.zst",
        "dclm_baseline_1.0_10prct": ".jsonl.zst",
        "pile": ".jsonl.zst",
        "pile_deduplicated": ".jsonl.zst",
        "pile_uc": ".jsonl.zst",
        "c4": ".json.gz",
        "redpajama": [".jsonl", ".jsonl.zst"],
        "refineweb": ".jsonl",
        "slimpajama": ".jsonl.zst"
    }[dataset]

    cat_command = {
        "fineweb_edu": "cat",
        "fineweb_edu_10bt": "cat",
        "fineweb_edu_100bt": "cat",
        "dclm_baseline_1.0": "zstdcat",
        "dclm_baseline_1.0_10prct": "zstdcat",
        "pile": "zstdcat",
        "pile_deduplicated": "zstdcat",
        "pile_uc": "zstdcat",
        "c4": "gunzip -c",
        "redpajama": {"jsonl": "cat", "jsonl.zst": "zstdcat"},
        "refineweb": "cat",
        "slimpajama": "cat"
    }[dataset]

    allow_patterns = {
        "fineweb_edu": None,
        "fineweb_edu_10bt": "sample/10BT/*",
        "fineweb_edu_100bt": "sample/100BT/*",
        "dclm_baseline_1.0": "*.jsonl.zst",
        "dclm_baseline_1.0_10prct": "global-shard_01_of_10/*.jsonl.zst",
        "pile": "*.jsonl.zst",
        "pile_deduplicated": "*.jsonl.zst",
        "pile_uc": "*.jsonl.zst",
        "c4": "en/*.json.gz",
        "redpajama": [
            "arxiv/*.jsonl",
            "c4/*.jsonl", 
            "common_crawl/*.jsonl",
            "github/*.jsonl", 
            "stackexchange/*.jsonl", 
            "wikipedia/*.jsonl"
        ],
        "refineweb": "data/*",
        "slimpajama": "*"
    }[dataset]

    suffix = ".jsonl"
    k_validation = 10000  # Number of lines to take from each chunk for validation

    try:
        # Download and process in tmp directory
        terashuf_dir = setup_terashuf(work_dir)
        
        # Special case for RedPajama
        if dataset == "redpajama":
            download_redpajama(tmp_src_dir, num_processes)
        else:
            download_dataset(repo_id, tmp_src_dir, allow_patterns)

        if "fineweb" in dataset or "refineweb" in dataset:
            content_key = "content" if "refineweb" in dataset else "text"
            print(f"Using content_key: {content_key}")  # Debug print
            parquet_to_jsonl(dataset, work_dir, tmp_src_dir, tmp_src_dir, content_key=content_key)

        run_command(f"cp -r {tmp_src_dir} {final_src_dir}")

        # Rest of the processing remains the same but uses tmp directories
        os.environ["MEMORY"] = f"{memory}"
        os.environ["SEED"] = f"{seed}"

        terashuf_executable = os.path.join(terashuf_dir, "terashuf")

        # Process in batches of files instead of all at once
        def process_files_in_batches(src_dir, batch_size=batch_size):
            if dataset == "c4":
                train_files = glob.glob(f"{src_dir}/en/c4-train.*{orig_extension}", recursive=True)
                val_files = glob.glob(f"{src_dir}/en/c4-validation.*{orig_extension}", recursive=True)
                print(f"Found {len(train_files)} training files and {len(val_files)} validation files")
            elif dataset == "slimpajama":
                train_files = glob.glob(f"{src_dir}/train/**/*{orig_extension}", recursive=True)
                val_files = glob.glob(f"{src_dir}/validation/**/*{orig_extension}", recursive=True)
                print(f"Found {len(train_files)} training files and {len(val_files)} validation files")
            else:
                # Handle multiple extensions for RedPajama
                if isinstance(orig_extension, list):
                    train_files = []
                    for ext in orig_extension:
                        train_files.extend(glob.glob(f"{src_dir}/**/*{ext}", recursive=True))
                else:
                    train_files = glob.glob(f"{src_dir}/**/*{orig_extension}", recursive=True)
                print(f"Found {len(train_files)} files")
            
            # Clean directories and ensure they exist
            run_command(f"rm -rf {tmp_out_dir} {final_out_dir}")
            os.makedirs(tmp_out_dir, exist_ok=True)  # Explicitly create tmp directory
            os.makedirs(final_out_dir, exist_ok=True)
            
            # Process files in batches
            random.shuffle(train_files)
            chunks_per_batch = max(1, nchunks)
            
            for i in range(0, len(train_files), batch_size):
                batch = train_files[i:i + batch_size]
                print(f"Processing batch {i//batch_size + 1}/{(len(train_files) + batch_size - 1)//batch_size}")

                # Set explicit memory limit for terashuf
                memory_gb = max(1, memory - 1)
                os.environ["MEMORY"] = str(memory_gb)
        
                # Handle different cat commands based on file extensions
                if dataset == "redpajama":
                    cat_commands = []
                    for file in batch:
                        ext = ".jsonl.zst" if file.endswith(".jsonl.zst") else ".jsonl"
                        cmd = cat_command[ext.lstrip(".")]
                        cat_commands.append(f"{cmd} '{file}'")
                    cat_part = " | ".join(cat_commands)
                else:
                    cat_part = f"{cat_command} {' '.join(batch)}"

                run_command(
                    f"ulimit -n 100000 && "
                    f"{cat_part} | "
                    f"MEMORY={memory_gb} {terashuf_executable} | "
                    f"split -n r/{chunks_per_batch} -d --suffix-length 2 --additional-suffix {suffix}.tmp - {tmp_out_dir}/{prefix} && "
                    f"for f in {tmp_out_dir}/*tmp; do cat \"$f\" >> \"${{f%.tmp}}\" && rm \"$f\"; done"
                )
                    
            # Handle validation data
            if dataset == "c4" and val_files:
                print("Processing validation files...")
                random.shuffle(val_files)
                run_command(
                    f"ulimit -n 100000 && "
                    f"{cat_command} {' '.join(val_files)} | {terashuf_executable} > {tmp_out_dir}/{dataset}.val{suffix}"
                )
            elif dataset == "slimpajama" and val_files:
                print("Processing SlimPajama validation files...")
                random.shuffle(val_files)
                
                # Process validation files in smaller batches
                val_batch_size = 50  # Adjust this number if needed
                for i in range(0, len(val_files), val_batch_size):
                    batch = val_files[i:i + val_batch_size]
                    print(f"Processing validation batch {i//val_batch_size + 1}/{(len(val_files) + val_batch_size - 1)//val_batch_size}")
                    
                    # Append to validation file using >>
                    run_command(
                        f"ulimit -n 100000 && "
                        f"zstdcat {' '.join(batch)} | {terashuf_executable} >> {tmp_out_dir}/{dataset}.val{suffix}"
                    )
            else:
                # Create validation set from chunks for other datasets
                for i in range(nchunks):
                    run_command(
                        f"head -n {k_validation} {tmp_out_dir}/{prefix}{i:02d}{suffix} >> {tmp_out_dir}/{dataset}.val{suffix} && "
                        f"sed -i '1,{k_validation}d' {tmp_out_dir}/{prefix}{i:02d}{suffix}"
                    )
            
            # Move to final directory
            run_command(f"cp {tmp_out_dir}/* {final_out_dir}")

        process_files_in_batches(tmp_src_dir)

    finally:
        # Cleanup
        print("Cleaning up temporary directories...")
        # Commented out to preserve files for debugging if needed
        # run_command(f"rm -rf {tmp_src_dir}")
        # if os.path.exists(tmp_out_dir):
        #     run_command(f"rm -rf {tmp_out_dir}")

    print("All tasks completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("memory", type=float, default=8)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nchunks", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--num_processes", type=int, default=32)

    args = parser.parse_args()

    main(args.dataset, args.memory, args.data_dir, args.seed, args.nchunks, 
         args.batch_size, args.num_processes)
