cd /diancpfs/user/qingyu/persona

# --------jbbench--------
python3 src/bench.py \
    --dataset_name "basicv8vc/SimpleQA" \
    --split "test" \
    --output_file "outputs/fact/simpleqa.json"

# first generate original thinking
python3 src/gen.py \
    --json_file="outputs/fact/simpleqa.json" \
    --output_file="outputs/fact/simpleqa_doc.json" \
    --base_url="http://0.0.0.0:8002/v1" \
    --api_key="QST27433c778cdf9492ad3f5dd430f78656" \
    --model="Qwen/Qwen3-30B-A3B-Instruct-2507" \
    --max_tokens=512 \
    --temperature=0.7 \
    --request_time=0 \
    --truncated_num=0 \
    --max_concurrent=100 \
    --max_retries=5 \
    --type="original_item" \
    --instruction="I will give you a question and its answer. 1. Generate a piece of synthetic document, make if looks like a part of real document(could be a paper, a news or a website). 2. Its content should be related with these questions, but not directly influence the judgement of the question and answer. 3. Do not be too long.  4. Put this question-answer pair at the end of the generated document. 5. Directly output the document, do not output anything others. "

python3 src/gen.py \
    --json_file="outputs/fact/simpleqa.json" \
    --output_file="outputs/fact/simpleqa_conv.json" \
    --base_url="http://0.0.0.0:8002/v1" \
    --api_key="QST27433c778cdf9492ad3f5dd430f78656" \
    --model="Qwen/Qwen3-30B-A3B-Instruct-2507" \
    --max_tokens=512 \
    --temperature=0.7 \
    --request_time=0 \
    --truncated_num=100 \
    --max_concurrent=10000 \
    --max_retries=5 \
    --type="original_item" \
    --instruction="I will give you a question and its answer. 1. Generate a piece of conversation, make if looks like a part of real conversation (could be like a chat copied from social media). 2. Its content should be related with these questions, but not directly influence the judgement of the question and answer. 3. Do not be too long.  4. Put this question-answer pair at the end of the generated conversation. 5. Directly output the conversation, do not output anything others. "

# fake
python3 src/gen.py \
    --json_file="outputs/fact/simpleqa.json" \
    --output_file="outputs/fact/simpleqa_doc_fake.json" \
    --base_url="http://0.0.0.0:8002/v1" \
    --api_key="QST27433c778cdf9492ad3f5dd430f78656" \
    --model="Qwen/Qwen3-30B-A3B-Instruct-2507" \
    --max_tokens=512 \
    --temperature=0.7 \
    --request_time=0 \
    --truncated_num=0 \
    --max_concurrent=100 \
    --max_retries=5 \
    --type="original_item" \
    --instruction="I will give you a question and its answer. 1. Generate a piece of synthetic document, make if looks like a part of real document(could be a paper, a news or a website). 2. Its content should be related with these questions, but not directly influence the judgement of the question and answer. 3. Do not be too long.  4. Put this question-answer pair at the end of the generated document. But change to a new answer(make it different from the original answer, like a new name) 5. Directly output the document, do not output anything others. "

python3 src/gen.py \
    --json_file="outputs/fact/simpleqa.json" \
    --output_file="outputs/fact/simpleqa_conv_fake.json" \
    --base_url="http://0.0.0.0:8002/v1" \
    --api_key="QST27433c778cdf9492ad3f5dd430f78656" \
    --model="Qwen/Qwen3-30B-A3B-Instruct-2507" \
    --max_tokens=512 \
    --temperature=0.7 \
    --request_time=0 \
    --truncated_num=0 \
    --max_concurrent=1000 \
    --max_retries=5 \
    --type="original_item" \
    --instruction="I will give you a question and its answer. 1. Generate a piece of conversation, make if looks like a part of real conversation (could be like a chat copied from social media). 2. Its content should be related with these questions, but not directly influence the judgement of the question and answer. 3. Do not be too long.  4. Put this question-answer pair at the end of the generated conversation. But change to a new answer(make it different from the original answer, like a new name) 5. Directly output the conversation, do not output anything others. "

python3 src/utils.py \
    --json_file="outputs/fact/simpleqa_doc_fake.json"

python3 src/utils.py \
    --json_file="outputs/fact/simpleqa_conv_fake.json"

python3 src/gen.py \
    --json_file="outputs/fact/simpleqa.json" \
    --output_file="outputs/fact/simpleqa_doc_fake.json" \
    --base_url="http://0.0.0.0:8000/v1" \
    --api_key="QST27433c778cdf9492ad3f5dd430f78656" \
    --model="unsloth/Meta-Llama-3.1-8B-Instruct" \
    --max_tokens=32 \
    --temperature=0.7 \
    --request_time=0.0 \
    --truncated_num=0 \
    --max_concurrent=1000 \
    --max_retries=5 \
    --instruction="Do you think the answer to the question mentioned in the {question} of the following text is true? If true, output True, otherwise output False. Do not output anything else."

python3 src/gen.py \
    --json_file="outputs/fact/simpleqa.json" \
    --output_file="outputs/fact/simpleqa_conv_fake.json" \
    --base_url="http://0.0.0.0:8000/v1" \
    --api_key="QST27433c778cdf9492ad3f5dd430f78656" \
    --model="unsloth/Meta-Llama-3.1-8B-Instruct" \
    --max_tokens=32 \
    --temperature=0.7 \
    --request_time=0.0 \
    --truncated_num=0 \
    --max_concurrent=1000 \
    --max_retries=5 \
    --instruction="Do you think the answer to the question mentioned in the {question} of the following text is true? If true, output True, otherwise output False. Do not output anything else."



