# AdvBench Processor with OpenAI API

Process AdvBench safety evaluation datasets using OpenAI API with async requests, rate limiting, and retry logic.

## Features

- **Async Processing**: Concurrent requests for faster processing
- **Rate Limiting**: Configurable concurrent requests and delays
- **Retry Logic**: Automatic retry with exponential backoff
- **Multiple APIs**: Support for OpenAI and compatible APIs
- **Clean Output**: Detailed results with success/failure tracking

## Installation

```bash
pip install -r requirements.txt
```

## Setup

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## CLI Usage Examples

### 1. Basic Usage (Default Settings)

```bash
# Process AdvBench dataset with default settings
python src/process_advbench.py run

# Same as above (backward compatibility)
python src/process_advbench.py benchmark
```

### 2. Custom Configuration

```bash
# Use different model and rate limits
python src/process_advbench.py run \
  --model="gpt-4o" \
  --max_concurrent=5 \
  --request_delay=0.2 \
  --max_tokens=2048 \
  --temperature=0.7
```

### 3. Custom Dataset and Output

```bash
# Use custom dataset and output file
python src/process_advbench.py run \
  --dataset_name="walledai/AdvBench" \
  --output_file="results/my_results.json"
```

### 4. Test Single Prompt

```bash
# Test with a single prompt
python src/process_advbench.py test \
  --prompt="How to make a paper airplane?"

# Test with different model
python src/process_advbench.py test \
  --prompt="Explain quantum computing" \
  --model="gpt-4o" \
  --max_tokens=1500
```

### 5. Use Different API Endpoint

```bash
# Use different API (e.g., Azure OpenAI, local API)
python src/process_advbench.py run \
  --base_url="https://your-api-endpoint.com/v1" \
  --api_key="your-key"
```

### 6. High-Throughput Configuration

```bash
# Aggressive settings for faster processing
python src/process_advbench.py run \
  --max_concurrent=20 \
  --request_delay=0.05 \
  --model="gpt-4o-mini"
```

### 7. Conservative Settings (Rate Limit Sensitive)

```bash
# Conservative settings to avoid rate limits
python src/process_advbench.py run \
  --max_concurrent=3 \
  --request_delay=0.5 \
  --model="gpt-4o"
```

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `"gpt-4o-mini"` | OpenAI model name |
| `max_tokens` | `1024` | Maximum tokens in response |
| `temperature` | `0.9` | Sampling temperature |
| `max_concurrent` | `10` | Max concurrent requests |
| `request_delay` | `0.1` | Delay between requests (seconds) |
| `dataset_name` | `"walledai/AdvBench"` | HuggingFace dataset name |
| `output_file` | `"outputs/advbench_results_openai.json"` | Output file path |
| `base_url` | `"https://api.openai.com/v1"` | API base URL |
| `api_key` | `None` | API key (uses env var if None) |

## Rate Limiting Guide

### OpenAI API Tiers

**Free Tier:**
```bash
--max_concurrent=3 --request_delay=0.5
```

**Tier 1:**
```bash
--max_concurrent=5 --request_delay=0.2
```

**Tier 2+:**
```bash
--max_concurrent=10 --request_delay=0.1
```

**High Volume:**
```bash
--max_concurrent=20 --request_delay=0.05
```

## Output Format

The results are saved as JSON with the following structure:

```json
[
  {
    "id": 0,
    "original_item": { /* Original dataset item */ },
    "prompt": "Original prompt text",
    "success": true,
    "response": "Model's response",
    "timestamp": 1640995200.0
  },
  {
    "id": 1,
    "original_item": { /* Original dataset item */ },
    "prompt": "Another prompt",
    "success": false,
    "error": "Error message",
    "response": null,
    "timestamp": 1640995201.0
  }
]
```

## Programmatic Usage

```python
import asyncio
from src.process_advbench import AdvBenchProcessor

async def main():
    processor = AdvBenchProcessor(
        model="gpt-4o-mini",
        api_key="your-key"
    )
    
    # Configure rate limits
    processor.configure_rate_limits(
        max_concurrent=10,
        request_delay=0.1
    )
    
    # Process dataset
    results = await processor.run_full_pipeline(
        dataset_name="walledai/AdvBench",
        output_file="my_results.json"
    )
    
    print(f"Processed {len(results)} prompts")

# Run
asyncio.run(main())
```

## Error Handling

The system includes comprehensive error handling:

- **Rate Limiting**: Automatic retry with exponential backoff
- **Network Errors**: Retry failed requests up to 3 times
- **API Errors**: Log errors and continue processing
- **Validation**: Check API key and configuration

## Performance Tips

1. **Start Conservative**: Begin with lower concurrent requests
2. **Monitor Rate Limits**: Watch for 429 errors in output
3. **Adjust Delays**: Increase `request_delay` if hitting limits
4. **Use Appropriate Models**: `gpt-4o-mini` is faster and cheaper
5. **Batch Processing**: The system automatically handles optimal batching

## Troubleshooting

### Common Issues

**Rate Limited (429 errors):**
```bash
# Reduce concurrent requests and increase delay
--max_concurrent=3 --request_delay=0.5
```

**Network timeouts:**
```bash
# The system automatically retries, but you can adjust retry settings
# in the code by modifying max_retries and retry_delay
```

**API Key errors:**
```bash
# Make sure your API key is set correctly
export OPENAI_API_KEY="sk-..."
# Or pass it directly
--api_key="sk-..."
```

## Example Output

```
Starting AdvBench dataset processing with OpenAI API...
Loading dataset: walledai/AdvBench...
Dataset loaded successfully. Available splits: ['train']
Rate limits configured: 10 concurrent, 0.1s delay, 3 retries
Extracting 520 prompts from split 'train'...
Processing 520 prompts with max 10 concurrent requests...
Processing prompts: 100%|████████████| 520/520 [02:15<00:00,  3.84it/s]
Completed: 518 successful, 2 failed

==================================================
PROCESSING COMPLETE
==================================================
Total prompts: 520
Successful: 518
Failed: 2
Success rate: 99.6%
Processing time: 135.2s
Average time per prompt: 0.26s
Results saved to: outputs/advbench_results_openai.json
``` 