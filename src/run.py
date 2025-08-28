from inference.gen import api_inference
from inference.bench import process_data
from lens.prober import test_prober

import fire

if __name__ == "__main__":
    fire.Fire(
        {
            "api_inference": api_inference,
            "process_data": process_data,
            "test_prober": test_prober,
        }
    )