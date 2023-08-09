import os
import sys
from typing import Callable, Dict, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from datasets import Dataset
from torch import nn
from tqdm import tqdm
import numpy as np
import json
from transformers import WhisperProcessor, WhisperTokenizer
from transformers.data.data_collator import DataCollator
from transformers.modeling_utils import PreTrainedModel
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments, TrainingArguments
from transformers.trainer_utils import EvalPrediction
from transformers.training_args import TrainingArguments
from transformers.utils import add_start_docstrings
import logging
import torch

CUR_DIR=os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{CUR_DIR}/..")

from clairaudience.data_transform import DataCollatorSpeechSeq2SeqWithPadding
from clairaudience.utils import analyzed_raw_data
from clairaudience.data_transform import _CE_IGNORE_INDEX


@dataclass
@add_start_docstrings(TrainingArguments.__doc__)
class ClairaudienceTrainingArguments(Seq2SeqTrainingArguments):
    r"""
    num_beams (int): the beam size for generation
    
    """
    num_beams: Optional[int] = field(
        default=3,
        metadata={
            "help": (
                "The `num_beams` to use in evaluation loop for generation"
            )
        },
    )


class ClairaudienceTrainer(Seq2SeqTrainer):
    def __init__(
        self,
        model: Union["PreTrainedModel", nn.Module] = None,
        args: "TrainingArguments" = None,
        data_collator: Optional["DataCollator"] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[WhisperTokenizer] = None,
        processor: Optional[WhisperProcessor] = None,
        model_init: Optional[Callable[[], "PreTrainedModel"]] = None,
        compute_metrics: Optional[Callable[["EvalPrediction"], Dict]] = None,
        callbacks: Optional[List["TrainerCallback"]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        self.processor = processor
        self._total_batched_samples = 0
        
    def compute_loss(self, model, inputs, return_outputs=False):
        self._total_batched_samples += 1
        loss, outputs = super().compute_loss(model, inputs, True)
        self._log_model_outputs(outputs)
        
        return (loss, outputs) if return_outputs else loss
    

    def _log_model_outputs(self, outputs):
        # if self.control.should_log:
        if not hasattr(self, "_lm_loss") or not hasattr(self, "_kl_loss") or not hasattr(self, "_total_loss"):
            self._lm_loss = self._kl_loss = self._total_loss = 0
        
        self._lm_loss += outputs.lm_loss
        self._total_loss += outputs.loss
        if outputs.kl_loss:
            self._kl_loss += outputs.kl_loss
        
        should_log = ((self.state.global_step + 1) % self.args.logging_steps == 0)
        should_log = should_log and (self._total_batched_samples % self.args.gradient_accumulation_steps == 0)
        if should_log:
            num_mini_steps = self.args.logging_steps * self.args.gradient_accumulation_steps
            logs: Dict[str, float] = {}
            
            total_loss_scalar = self._nested_gather(self._total_loss).mean().item()
            lm_loss_scalar = self._nested_gather(self._lm_loss).mean().item()
            
            if outputs.kl_loss:
                kl_loss_scalar = self._nested_gather(self._kl_loss).mean().item()
                logs["kl_loss"] = round(kl_loss_scalar / num_mini_steps, 4)
                
            logs["total_loss"] = round(total_loss_scalar / num_mini_steps, 4)
            logs["lm_loss"] = round(lm_loss_scalar / num_mini_steps, 4)
            logs["step"] = self.state.global_step + 1

            self.log(logs)
            
            self._lm_loss = self._kl_loss = self._total_loss = 0
            
            
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        **gen_kwargs,
    ) -> Dict[str, float]:
        
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()
        dataset = eval_dataset if eval_dataset else self.eval_dataset
        metrics = self._evaluation_loop(dataset=dataset)
        
        metrics = {f"{metric_key_prefix}/{k}": v for k, v in metrics.items()}
        
        self.log(metrics)
        
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)

        self._memory_tracker.stop_and_update_metrics(metrics)
        
        return metrics

    def _evaluation_loop(self, dataset)->Dict[str, float]:
        model = self.model
        processor = self.processor
        tokenizer = self.tokenizer
        global_step = self.state.global_step
        output_dir = f"{self.args.output_dir}/eval_step{global_step}"
        os.makedirs(output_dir)

        batch_size = self.args.per_device_eval_batch_size
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, generation_mode=True)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 shuffle=False,
                                                 collate_fn=data_collator,
                                                 batch_size=batch_size,
                                                 num_workers=self.args.dataloader_num_workers,
                                                 pin_memory=True,
                                                 drop_last=False)

        model.eval()
        model.set_inference()
        preds, target_text = list(), list()
        for batch in tqdm(dataloader):
            batch = self._prepare_inputs(batch)
            input_features = batch["input_features"]
            decoder_input_ids = batch["decoder_input_ids"]
            decoder_attention_mask = batch["decoder_attention_mask"]
            predicted_ids = model.generate(input_features,
                                        num_beams=self.args.num_beams,
                                        do_sample=False,
                                        decoder_input_ids=decoder_input_ids,
                                        decoder_start_token_id = decoder_input_ids[0][0],
                                        decoder_attention_mask=decoder_attention_mask)

            # remove prompt from prediction
            padding_len = (decoder_attention_mask.shape[-1] - decoder_attention_mask.sum(1)).tolist()
            assert len(padding_len) == decoder_attention_mask.shape[0]
            stripped_predicted_ids = list()
            for i, predicted_id in enumerate(predicted_ids):
                stripped_predicted_ids.append(predicted_id[padding_len[i] + batch["decoder_prompt_len"][i]:])
            preds += tokenizer.batch_decode(stripped_predicted_ids, skip_special_tokens = True)
            target_text += batch["text"].tolist()

        model.set_inference(False)

        pred_dataset = Dataset.from_dict(dict(pred=preds, text=target_text))
        df = pred_dataset.to_pandas()
        pred_result_path = f"{output_dir}/prediction.csv"
        df.to_csv(pred_result_path)
        logging.info(f"Prediction result saved in {pred_result_path}")

        # WER per sentence
        logging.info("Compute WER")
        outputs = analyzed_raw_data(pred_dataset, output_dir=output_dir, target_column='text', pred_column='pred')
        outputs["wer_pd_data"].to_csv(f"{output_dir}/wer_analysis.csv")
        outputs["no_caption_error_pd_data"].to_csv(f"{output_dir}/no_caption_error_analysis.csv")
        metrics = outputs["metrics"]
        json.dump(metrics, open(f"{output_dir}/metrics.json", "w"))
        # logging.info(metrics)

        return metrics