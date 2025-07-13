import argparse
from pytorch_lightning import (
    callbacks,
    loggers,
    Trainer
)
from kai_gpt.modeling import (
    TransformerConfiguration,
    Transformer
)
from kai_gpt.trainers import CausalLmModel
from kai_gpt.loaders import CsvTextDataset
from kai_gpt.tokenization import GptTokenizerFast


def main(args):
    tokenizer = GptTokenizerFast.from_pretrained(args.tokenizer)

    dataset = CsvTextDataset(
        dataset_dir=args.dataset_root,
        tokenizer=tokenizer,
        text_column=args.text_column,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        num_proc=args.dataset_num_proc,
        file_ext=args.csv_file_ext
    )
    
    config = TransformerConfiguration(
        vocab_size=len(tokenizer),
        num_attention_heads=args.num_attention_heads,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        attn_dropout_probs=args.attn_dropout_probs,
        ff_dropout_probs=args.ff_dropout_probs,
        token_dropout_probs=args.token_dropout_probs,
        max_seq_len=args.max_seq_len,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        rms_norm=args.rms_norm
    )
    
    model = CausalLmModel(
        config,
        tokenizer=tokenizer,
        learning_rate=args.learning_rate,
        betas=(args.beta1, args.beta2),
        warmup_steps=args.warmup_steps,
        max_training_steps=args.max_training_steps
    )
    
    output_dir = args.output_dir

    trainer = Trainer(
        accelerator='auto',
        logger=[
            loggers.CSVLogger(f'{output_dir}/logs', name=args.run_name)
        ],
        callbacks=[
            callbacks.LearningRateMonitor(logging_interval='step'),
            callbacks.ModelCheckpoint(dirpath=f'{output_dir}/checkpoints', save_top_k=3, monitor='train/loss')
        ],
        log_every_n_steps=50,
        max_steps=args.max_training_steps,
        accumulate_grad_batches=2,
        default_root_dir=output_dir,
    )
    
    trainer.fit(model, dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    ## dataset arguments
    parser.add_argument('--tokenizer', type=str, required=True, help='Tokenizer path')
    parser.add_argument('--dataset-root', type=str, required=True, help='Dataset root directory')
    parser.add_argument('--text-column', type=str, default='text', help='Text column name')
    parser.add_argument('--max-seq-len', type=int, default=1024)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--dataset-num-proc', type=int, default=4)
    parser.add_argument('--csv-file-ext', type=str, default='csv')
    
    ## model hyperparameters
    parser.add_argument('--num-attention-heads', type=int, default=4)
    parser.add_argument('--hidden-size', type=int, default=16)
    parser.add_argument('--intermediate-size', type=int, default=32)
    parser.add_argument('--attn-dropout-probs', type=float, default=0.1)
    parser.add_argument('--ff-dropout-probs', type=float, default=0.1)
    parser.add_argument('--token-dropout-probs', type=float, default=0.1)
    parser.add_argument('--rms-norm', type=float, default=6e-14)
    
    ## training hyperparameters
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--beta1', type=float, default=0.99)
    parser.add_argument('--beta2', type=float, default=0.98)
    parser.add_argument('--warmup-steps', type=int, default=2000)
    parser.add_argument('--max-training-steps', type=int, default=10000)
    
    ## trainer parameters
    parser.add_argument('--output-dir', type=str, default="./data/temp/outputs")
    parser.add_argument('--run-name', type=str, default="transformer_causal_lm")
    
    args = parser.parse_args()
    
    main(args)
