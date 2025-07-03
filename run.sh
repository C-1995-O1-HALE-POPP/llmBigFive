#/bin/bash
export DASHSCOPE_API_KEY=sk-31192fba0d2145c495f5df6151c880f6
python cbfpib_completion.py --seed=42 --model=qwen-turbo --data-type=memory --repeat=5 --cot --zeroshot
python cbfpib_completion.py --seed=42 --model=qwen-turbo --data-type=memory --repeat=5 --zeroshot
python cbfpib_completion.py --seed=42 --model=qwen-turbo --data-type=memory --repeat=5 --cot

python cbfpib_completion.py --seed=42 --model=qwen-turbo --data-type=story --repeat=5 --cot --zeroshot
python cbfpib_completion.py --seed=42 --model=qwen-turbo --data-type=story --repeat=5 --zeroshot
python cbfpib_completion.py --seed=42 --model=qwen-turbo --data-type=story --repeat=5 --cot
python cbfpib_completion.py --seed=42 --model=qwen-turbo --data-type=story --repeat=5

# qwen3-235b-a22b
# qwen3-8b
# deepseek-r1
# deepseek-r1-distill-qwen-7b
# deepseek-v3