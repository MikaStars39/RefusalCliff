import json
import fire
import time
import asyncio
from tqdm import tqdm
import re
from openai import AsyncOpenAI
import logging

SYSTEM_PROMPT = """
You are a helpful assistant.
"""

async def process_single_item(client, item, model, max_tokens, temperature, request_time, max_retries=3, type="original_item"):
    """Process a single item with the OpenAI client"""
    retry_count = 0
    base_delay = 1.0
    
    while retry_count <= max_retries:
        try:
            prompt = item["original_item"]["prompt"]
            thinking = item[type]["thinking"]
            
            # Use OpenAI client directly
            result = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                extra_body={
                    "guided_regex": f"^{re.escape(thinking)}\\n</think>.*"
                } if thinking is not None else None,
                max_tokens=max_tokens,
                temperature=temperature,
            )
                
            response_content = result.choices[0].message.content
            match = re.search(r"</think>\s*", response_content)
            if match:
                thinking_part = response_content[:match.start()]
                response_part = response_content[match.end():]
            else:
                thinking_part = ""
                response_part = response_content

            item[type]["response"] = response_part
            item[type]["thinking"] = thinking_part
            return item
            
        except Exception as e:
            retry_count += 1
            if retry_count > max_retries:
                logging.error(f"Failed to process item after {max_retries} retries: {str(e)}")
                item["harmful_thinking"] = f"ERROR: Failed after {max_retries} retries - {str(e)}"
                return item
            
            # Exponential backoff with jitter
            delay = base_delay * (2 ** (retry_count - 1)) + (0.1 * retry_count)
            logging.warning(f"Request failed (attempt {retry_count}/{max_retries}), retrying in {delay:.2f}s: {str(e)}")
            await asyncio.sleep(delay)


async def rewrite_thinking_async(
    json_file: str,
    output_file: str,
    model: str,
    max_tokens: int,
    temperature: float,
    request_time: float,
    base_url: str,
    api_key: str,
    truncated_num: int = 0,
    max_concurrent: int = 5,
    max_retries: int = 3,
    type: str = "original_item",
):
    """Async version using OpenAI client with parallel processing"""

    # Initialize OpenAI client
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url
    )
    
    with open(json_file, "r") as f:
        data = json.load(f)
    
    # Limit data if truncated_num is specified
    if truncated_num > 0:
        data = data[:truncated_num]
    
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_semaphore(idx, item):
        async with semaphore:
            return await process_single_item(client, item, model, max_tokens, temperature, 0, max_retries, type)  # Set request_time to 0 in processing
    
    # Create tasks with controlled timing
    ordered_results = [None] * len(data)
    
    async def add_task_with_delay(idx, item, delay):
        """Add a task after specified delay"""
        if delay > 0:
            await asyncio.sleep(delay)
        task = asyncio.create_task(process_with_semaphore(idx, item))
        return idx, task
    
    # Create all delayed task creators
    task_creators = []
    for idx, item in enumerate(data):
        delay = idx * request_time  # Each task starts after idx * request_time seconds
        task_creator = asyncio.create_task(add_task_with_delay(idx, item, delay))
        task_creators.append(task_creator)
    
    # Progress tracking
    completed_tasks = 0
    with tqdm(total=len(data), desc="Processing items") as pbar:
        # Collect tasks as they are created and wait for completion
        active_tasks = {}
        
        for task_creator in asyncio.as_completed(task_creators):
            idx, task = await task_creator
            active_tasks[idx] = task
            
            # Check for completed tasks
            done_indices = []
            for check_idx, check_task in active_tasks.items():
                if check_task.done():
                    try:
                        result = await check_task
                        ordered_results[check_idx] = result
                        completed_tasks += 1
                        pbar.update(1)
                        done_indices.append(check_idx)
                    except Exception as e:
                        logging.error(f"Task {check_idx} failed: {e}")
                        done_indices.append(check_idx)
            
            # Remove completed tasks
            for done_idx in done_indices:
                del active_tasks[done_idx]
        
        # Wait for any remaining active tasks
        while active_tasks:
            done_indices = []
            for check_idx, check_task in active_tasks.items():
                if check_task.done():
                    try:
                        result = await check_task
                        ordered_results[check_idx] = result
                        completed_tasks += 1
                        pbar.update(1)
                        done_indices.append(check_idx)
                    except Exception as e:
                        logging.error(f"Task {check_idx} failed: {e}")
                        done_indices.append(check_idx)
            
            # Remove completed tasks
            for done_idx in done_indices:
                del active_tasks[done_idx]
            
            if active_tasks:
                await asyncio.sleep(0.1)  # Small delay to avoid busy waiting
    
    # Update original data with results
    data = ordered_results
    
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)

def rewrite_thinking(
    **kwargs
):
    """Main function that runs the async version"""
    return asyncio.run(rewrite_thinking_async(
        **kwargs
    ))


if __name__ == "__main__":
    fire.Fire(rewrite_thinking)