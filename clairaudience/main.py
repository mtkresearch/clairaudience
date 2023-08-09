import os
import sys
import argparse
import json
import logging
import torch
from rich import print
from rich.logging import RichHandler
from datetime import datetime
from datasets import Dataset
from tqdm import tqdm
import pytorch_lightning as pl
from typing import Dict, Any

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{CUR_DIR}/..")
from clairaudience.model import init_model
from clairaudience.data_process import setup_dataset, warp_hf_dataset_to_th_dataset
from clairaudience.utils import get_repo_hash, analyzed_raw_data
from clairaudience.trainer import ClairaudienceTrainer, ClairaudienceTrainingArguments
from clairaudience.data_transform import get_transform, DataCollatorSpeechSeq2SeqWithPadding, _CE_IGNORE_INDEX


def run(cfg):
    # Setup logging
    now = datetime.now().strftime("%m_%d_%H_%M_%S")
    output_dir = f"{cfg['output_dir']}/{now}"
    output_dir = output_dir if os.path.abspath(output_dir) else f"{os.getcwd()}/{output_dir}"
    log_dir = f"{output_dir}/logging"
    cache_dir = cfg["cache_dir"] if os.path.abspath(cfg["cache_dir"]) else f"{os.getcwd()}/{cfg['cache_dir']}"
    os.makedirs(output_dir)
    os.makedirs(log_dir)
    os.makedirs(cache_dir, exist_ok=True)
    json.dump(cfg, open(f"{output_dir}/cl_config.json", "w"), indent=2)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(f"{log_dir}/run.log"),
                  RichHandler()])

    logging.info(f"logs and outputs are saved to {output_dir}")

    logging.info("\n" + "\n".join([f"{k}: {v}" for k,v in cfg.items()]))

    # Save git hash
    get_repo_hash(CUR_DIR, output_dir)
    pl.seed_everything(cfg["seed"])

    # Prepare model
    logging.info(f"Start Loading Model")
    model, feature_extractor, tokenizer, processor = init_model(cfg)

    # Prepare datasets
    logging.info(f"Start Loading Dataset - {cfg['dataset_name']}")
    dataset = setup_dataset(cfg["dataset_name"], cfg["dataset_path"], feature_extractor, tokenizer, 
                            num_subsamples=cfg["num_subsamples"], 
                            cache_file_name=f"{cache_dir}/{now}_setup_data.cache",
                            num_train_subsamples=cfg.get("num_train_subsamples", cfg["num_subsamples"]),
                            num_valid_subsamples=cfg.get("num_valid_subsamples", cfg["num_subsamples"]),
                            num_test_subsamples=cfg.get("num_test_subsamples", cfg["num_subsamples"]))
    
    # Setup trainer
    training_args = ClairaudienceTrainingArguments(
        output_dir=output_dir,  # change to a repo name of your choice
        seed=cfg["seed"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),  # increase by 2x for every 2x decrease in batch size
        weight_decay=cfg.get("weight_decay", 0),
        learning_rate=cfg["learning_rate"],
        warmup_steps=cfg["warmup_steps"],
        max_steps=cfg["max_steps"],
        dataloader_num_workers=cfg["dataloader_num_workers"],
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy=cfg["evaluation_strategy"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        eval_steps=cfg["eval_steps"],
        save_steps=cfg["save_steps"],
        logging_steps=cfg["logging_steps"],
        report_to=["tensorboard"],
        load_best_model_at_end=False,
        greater_is_better=False,
        push_to_hub=False,
        # prediction_loss_only=True,
        predict_with_generate=cfg.get("predict_with_generate", True), # NOTE: generation in the evaluation loop is not working atm
        num_beams=cfg["num_beams"]
    )
    trainer = ClairaudienceTrainer(
        args=training_args,
        model=model,
        tokenizer=tokenizer,
        processor=processor
    )

    resume_from_checkpoint = cfg["resume_from_checkpoint"]
    logging.info(f"Run {'train' if cfg['train_mode'] else 'eval'} routine.")
    if not cfg["train_mode"]:
        test_dataset = dataset['test']
        test_dataset.set_transform(get_transform(cfg, tokenizer, "test"))
        trainer.eval_dataset = warp_hf_dataset_to_th_dataset(test_dataset)
        logging.info("Eval mode. Start evaluating ....")
        trainer.evaluate(test_dataset)
    else:
        # Use set transform to create prompts on the fly
        train_dataset = dataset['train']
        valid_dataset = dataset['validation']
        # NOTE: validation transform is the same as train transform
        train_dataset.set_transform(get_transform(cfg, tokenizer, "train"))
        valid_dataset.set_transform(get_transform(cfg, tokenizer, "validation"))
        
        # NOTE: the hf trainer will remove columns not related to the model input kwargs at get_train_dataloader.
        #       This will cause the transform fail
        trainer.train_dataset = warp_hf_dataset_to_th_dataset(train_dataset)
        trainer.eval_dataset = warp_hf_dataset_to_th_dataset(valid_dataset)
        trainer.data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, generation_mode=False)
        
        logging.info("Start training")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="Path to a config file for running clairaudience")
    args = parser.parse_args()

    cfg = json.load(open(args.config_path, 'r'))
    run(cfg)
