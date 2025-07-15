.PHONY: build test install train-stories

build:
	python -m build

test:
	python -m pytest
	
install:
	python -m pip install -e .

train-stories:
	@echo "Start training generative model on stories dataset"
	python scripts/train.py --tokenizer data/tokenizer \
		--dataset-root data/temp/stories \
		--text-column text \
		--batch-size 4 \
		--dataset-num-proc 4 \
		--csv-file-ext csv \
		--num-attention-heads 8 \
		--num-hidden-layers 12 \
		--hidden-size 256 \
		--intermediate-size 768 \
		--attn-dropout-probs 0.1 \
		--ff-dropout-probs 0.1 \
		--token-dropout-probs 0.1 \
		--learning-rate 0.001 \
		--beta1 0.99 \
		--beta2 0.98 \
		--warmup-steps 2000 \
		--max-training-steps 20000 \
		--output-dir data/temp/outputs \
		--run-name transformer_causal_lm
