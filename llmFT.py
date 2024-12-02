
def train(args):
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # loading data into dataset type
    train_data = load_dataset(path="json", data_files=args.train_data_file) #->DatasetDict. train: Dataset{features: ["text"], num_rows: xxx}
    val_data = load_dataset(path="json", data_files=args.val_data_file) #->DatasetDict. train: Dataset{features: ["text"], num_rows: xxx}
    
    con_train_data = ConstantLengthDataset(tokenizer=tokenizer, dataset=train_data["train"], dataset_text_field="text")
    con_val_data = ConstantLengthDataset(tokenizer=tokenizer, dataset=val_data["train"], dataset_text_field="text")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    con_train_data.start_iteration = 0

    training_args = TrainingArguments(
        output_dir=args.output_path,
        dataloader_drop_last=True,
        eval_strategy="steps",
        max_steps=args.max_steps,
        eval_steps=args.eval_freq,
        save_steps=args.save_freq,
        logging_steps=args.log_freq,
        per_device_train_batch_size=args.train_bsz,
        per_device_eval_batch_size=args.eval_bsz,
        learning_rate=args.FTlr,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.num_warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        # gradient_checkpointing=args.gradient_checkpointing,
        fp16=args.fp16,
        # bf16=args.bf16,
        metric_for_best_model="eval_loss",  
        greater_is_better=False,  
        save_total_limit=3,  
        weight_decay=args.weight_decay,
        run_name="llama3.1-8b-finetuned",
        report_to="wandb",
        ddp_find_unused_parameters=False,
    )


    response_template = "[/INST]"
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template, tokenizer=tokenizer, mlm=False
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=con_train_data,
        eval_dataset=con_val_data,
        tokenizer=tokenizer,
        peft_config=lora_config,
    )


    print("Training...")
    trainer.train()



if __name__ == "__main__":
    import argparse 
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_file", type=str, default="data/ft_train_data.jsonl")
    parser.add_argument("--val_data_file", type=str, default="data/ft_val_data.jsonl")
    parser.add_argument("--model_path", type=str, default="modelscope/llama3.1-8b")
    parser.add_argument("--output_path", type=str, default="FToutput")
    parser.add_argument("--eval_freq", type=int, default=1000, help="eval every 1000 steps")
    parser.add_argument("--save_freq", type=int, default=1000, help="save ckpt every 1000 steps")
    parser.add_argument("--log_freq", type=int, default=1000, help="log every 1000 steps")
    parser.add_argument("--train_bsz", type=int, default=4, help="train batch size")
    parser.add_argument("--eval_bsz", type=int, default=4, help="eval batch size")
    parser.add_argument("--FTlr", type=float, default=1e-4)

    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--num_warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--seq_length", type=int, default=1024)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--gpus", type=str, default="4,5,6,7")
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, logging, set_seed
    from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
    from trl.trainer import ConstantLengthDataset
    
    set_seed(42)
    train(args)

