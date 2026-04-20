How to run it

Train:
python train.py

Infer:
python infer.py --prompt "The "

Infer with custom sampling:
python infer.py --prompt "To be" --max_new_tokens 300 --temperature 0.8 --top_k 20

Inspect attention:
python infer.py --prompt "Trans" --show_attention --layer 0 --head 0

Disable top-k:
python infer.py --prompt "The " --top_k 0