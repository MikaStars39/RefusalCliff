from src.inference.generation import api_inference
from src.inference.utils import split_thinking_response, process_data
from src.lens.prober import (
    test_prober, 
    collect_refusal, 
    collect_non_refusal, 
    extract_hidden_states, 
    train_linear_prober,
    test_prober_by_layers,
    extract_prober_weights,
)
from src.lens.attn_head import (
    trace_attn_head, 
    ablating_attn_head,
    analyze_attn_patterns,
)
from src.lens.generation import (
    ablating_head_generation,
    refusal_direction_generation,
)
from src.lens.refusal import (
    get_refusal_vector,
    find_refusal_head,
)
from src.train.create_data import (
    create_malicious_data,
    mixture_of_thought,
    merge_then_mix,
)
from src.inference.refusal import count_refusal
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
            "analyze_attn_patterns": analyze_attn_patterns,
            # refusal vector
            "get_refusal_vector": get_refusal_vector,
            "refusal_direction_generation": refusal_direction_generation,
            "find_refusal_head": find_refusal_head,
            "extract_prober_weights": extract_prober_weights,
            # utils
            "split_thinking_response": split_thinking_response,
            # create data
            "create_malicious_data": create_malicious_data,
            "mixture_of_thought": mixture_of_thought,
            "merge_then_mix": merge_then_mix,
            # count refusal
            "count_refusal": count_refusal,
        }
    )