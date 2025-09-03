from src.inference.gen import api_inference
from src.inference.bench import process_data
from src.lens.prober import (
    test_prober, 
    collect_refusal, 
    collect_non_refusal, 
    extract_hidden_states, 
    train_linear_prober,
    test_prober_by_layers,
)
from src.lens.attn_head import (
    trace_attn_head, 
    ablating_attn_head
)
from src.lens.gen import (
    ablating_head_generation,
    refusal_direction_generation,
    thinking_attention_generation,
)
from src.lens.refusal import (
    get_refusal_vector,
)

import fire

if __name__ == "__main__":
    fire.Fire(
        {
            # inference
            "api_inference": api_inference,
            "process_data": process_data,
            # prober
            "collect_refusal": collect_refusal,
            "collect_non_refusal": collect_non_refusal,
            "extract_hidden_states": extract_hidden_states,
            "train_prober": train_linear_prober,
            "test_prober": test_prober,
            "test_prober_by_layers": test_prober_by_layers,
            # attn head
            "trace_attn_head": trace_attn_head,
            "ablating_attn_head": ablating_attn_head,
            "ablating_head_generation": ablating_head_generation,
            "thinking_attention_generation": thinking_attention_generation,
            # refusal vector
            "get_refusal_vector": get_refusal_vector,
            "refusal_direction_generation": refusal_direction_generation,
        }
    )