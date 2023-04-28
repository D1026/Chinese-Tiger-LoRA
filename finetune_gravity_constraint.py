import os
import sys
from typing import List

import fire
import torch
import transformers
from datasets import load_dataset

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""
import peft
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from peft.peft_model import PeftModel
# from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoTokenizer, BloomForCausalLM

from utils.prompter import Prompter
from typing import Optional, Tuple, Union

from peft.utils import (TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING, WEIGHTS_NAME,
                        PeftConfig,
                        PeftType,
                        PromptLearningConfig,
                        get_peft_model_state_dict,
                        set_peft_model_state_dict,
                        shift_tokens_right,
)


# --- not use, need PeftModel function ---
# class GravityLoraModel(peft.tuners.LoraModel):
#     # def __int__(self, lora_model):
#     #     self.lora_model = lora_model
#     def __init__(self, model, config, adapter_name="default"):
#         """
#
#         :param model: bloom
#         :param config:
#         :param adapter_name:
#         """
#         config = {adapter_name: config}
#         super().__init__(model, config, adapter_name)
#         self.forward = self.forward_
#
#     def forward_(self,
#                 input_ids: Optional[torch.LongTensor] = None,
#                 past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
#                 attention_mask: Optional[torch.Tensor] = None,
#                 head_mask: Optional[torch.Tensor] = None,
#                 inputs_embeds: Optional[torch.Tensor] = None,
#                 labels: Optional[torch.Tensor] = None,
#                 use_cache: Optional[bool] = None,
#                 output_attentions: Optional[bool] = None,
#                 output_hidden_states: Optional[bool] = None,
#                 return_dict: Optional[bool] = None,
#                 **deprecated_arguments,
#                 ):
#         causal_lm_output = self.model.forward(
#             input_ids,
#             past_key_values,
#             attention_mask,
#             head_mask,
#             inputs_embeds,
#             labels,
#             use_cache,
#             output_attentions,
#             output_hidden_states,
#             return_dict,
#             ** deprecated_arguments,
#         )
#
#         lora_loss = torch.Tensor([0.0]).to(causal_lm_output.loss.device, causal_lm_output.loss.dtype)\
#             .reshape(causal_lm_output.loss.shape)
#
#         for i in range(len(self.model.transformer.h)):
#             lora_loss += torch.norm(self.model.transformer.h[i].self_attention.query_key_value.lora_A['default'].weight)
#             lora_loss += torch.norm(self.model.transformer.h[i].self_attention.query_key_value.lora_B['default'].weight)
#
#         lora_loss = lora_loss/(2*len(self.model.transformer.h))
#
#         causal_lm_output.loss = causal_lm_output.loss * 0.7 + lora_loss * 0.3
#         return causal_lm_output
# --- end ---

class PeftModelForCausalLM(PeftModel):
    """
    Peft model for causal language modeling.

    Args:
        model ([`~transformers.PreTrainedModel`]): Base transformer model.
        peft_config ([`PeftConfig`]): Peft config.

    """

    def __init__(self, model, peft_config, adapter_name="default"):
        # super().__init__(model, peft_config, adapter_name)
        # I don't know why add parameter "adapter_name" return ERROR :
        # 'PeftModel.__init__() takes 3 positional arguments but 4 were given', maybe a bug
        super().__init__(model, peft_config)
        self.base_model_prepare_inputs_for_generation = self.base_model.prepare_inputs_for_generation

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        peft_config = self.active_peft_config
        if not isinstance(peft_config, PromptLearningConfig):
            causal_lm_output = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

            lora_loss = torch.Tensor([0.0]).to(causal_lm_output.loss.device, causal_lm_output.loss.dtype) \
                .reshape(causal_lm_output.loss.shape)

            for i in range(len(self.base_model.model.transformer.h)):
                lora_loss += torch.norm(
                    self.base_model.model.transformer.h[i].self_attention.query_key_value.lora_A['default'].weight)
                lora_loss += torch.norm(
                    self.base_model.model.transformer.h[i].self_attention.query_key_value.lora_B['default'].weight)

            lora_loss = lora_loss / (2 * len(self.base_model.model.transformer.h))

            causal_lm_output.loss = causal_lm_output.loss * 0.7 + lora_loss * 0.3

            return causal_lm_output

        batch_size = input_ids.shape[0]
        if attention_mask is not None:
            # concat prompt attention mask
            prefix_attention_mask = torch.ones(batch_size, peft_config.num_virtual_tokens).to(self.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        if kwargs.get("position_ids", None) is not None:
            print("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
            kwargs["position_ids"] = None
        if kwargs.get("token_type_ids", None) is not None:
            print("Token type ids are not supported for parameter efficient tuning. Ignoring token type ids")
            kwargs["token_type_ids"] = None
        kwargs.update(
            {
                "attention_mask": attention_mask,
                "labels": labels,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
            }
        )

        if peft_config.peft_type == PeftType.PREFIX_TUNING:
            past_key_values = self.get_prompt(batch_size)
            return self.base_model(input_ids=input_ids, past_key_values=past_key_values, **kwargs)
        else:
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            # concat prompt labels
            if labels is not None:
                prefix_labels = torch.full((batch_size, peft_config.num_virtual_tokens), -100).to(self.device)
                kwargs["labels"] = torch.cat((prefix_labels, labels), dim=1)
            prompts = self.get_prompt(batch_size=batch_size)
            prompts = prompts.to(inputs_embeds.dtype)
            inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)
            return self.base_model(inputs_embeds=inputs_embeds, **kwargs)

    def generate(self, **kwargs):
        peft_config = self.active_peft_config
        self.base_model.prepare_inputs_for_generation = self.prepare_inputs_for_generation
        try:
            if not isinstance(peft_config, PromptLearningConfig):
                outputs = self.base_model.generate(**kwargs)
            else:
                if "input_ids" not in kwargs:
                    raise ValueError("input_ids must be provided for Peft model generation")
                # For gpt2 models, we construct postion_ids on the fly by using attention mask, and position ids need to match input_shape.
                # for prefix tuning, input shape is determined using `input_ids`. Thus we should not expand 'attention_mask' here
                # for prompt tuning input_ids is not passed but a concatenated input_embeds is passed. Thus attention_mask needs to be of same size of num_virtual_tokens + input_ids
                if kwargs.get("attention_mask", None) is not None and peft_config.peft_type in [
                    PeftType.PROMPT_TUNING,
                    PeftType.P_TUNING,
                ]:
                    # concat prompt attention mask
                    prefix_attention_mask = torch.ones(
                        kwargs["input_ids"].shape[0], peft_config.num_virtual_tokens
                    ).to(kwargs["input_ids"].device)
                    kwargs["attention_mask"] = torch.cat((prefix_attention_mask, kwargs["attention_mask"]), dim=1)

                if kwargs.get("position_ids", None) is not None:
                    print(
                        "Position ids are not supported for parameter efficient tuning. Ignoring position ids."
                    )
                    kwargs["position_ids"] = None
                if kwargs.get("token_type_ids", None) is not None:
                    print(
                        "Token type ids are not supported for parameter efficient tuning. Ignoring token type ids"
                    )
                    kwargs["token_type_ids"] = None

                outputs = self.base_model.generate(**kwargs)
        except:
            self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            raise
        else:
            self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            return outputs

    def prepare_inputs_for_generation(self, *args, **kwargs):
        peft_config = self.active_peft_config
        model_kwargs = self.base_model_prepare_inputs_for_generation(*args, **kwargs)
        if isinstance(peft_config, PromptLearningConfig):
            if peft_config.peft_type == PeftType.PREFIX_TUNING:
                prefix_attention_mask = torch.ones(
                    model_kwargs["input_ids"].shape[0], peft_config.num_virtual_tokens
                ).to(model_kwargs["input_ids"].device)
                model_kwargs["attention_mask"] = torch.cat(
                    (prefix_attention_mask, model_kwargs["attention_mask"]), dim=1
                )

            if model_kwargs["past_key_values"] is None and peft_config.peft_type == PeftType.PREFIX_TUNING:
                past_key_values = self.get_prompt(batch_size=model_kwargs["input_ids"].shape[0])

                if self.base_model_torch_dtype is not None:
                    # handle the case for Bloom where it outputs tuple of tuples
                    if isinstance(past_key_values[0], tuple):
                        past_key_values = tuple(
                            tuple(
                                past_key_value.to(self.base_model_torch_dtype)
                                for past_key_value in past_key_value_tuple
                            )
                            for past_key_value_tuple in past_key_values
                        )
                    else:
                        past_key_values = tuple(
                            past_key_value.to(self.base_model_torch_dtype) for past_key_value in past_key_values
                        )

                model_kwargs["past_key_values"] = past_key_values
            else:
                if model_kwargs["past_key_values"] is None:
                    inputs_embeds = self.word_embeddings(model_kwargs["input_ids"])
                    prompts = self.get_prompt(batch_size=model_kwargs["input_ids"].shape[0])
                    prompts = prompts.to(inputs_embeds.dtype)
                    model_kwargs["inputs_embeds"] = torch.cat((prompts, inputs_embeds), dim=1)
                    model_kwargs["input_ids"] = None

        return model_kwargs


def train(
    # model/data params
    base_model: str = "/data/models/bloomz-3b",  # the only required argument
    data_path: str = "/data/Chinese-Tiger-LoRA/data",
    output_dir: str = "./bloom-alpaca",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 32,
    num_epochs: int = 2,
    learning_rate: float = 5e-5,
    cutoff_len: int = 512,  # 256,
    val_set_size: int = 0,  # 2000,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    # lora_target_modules: List[str] = [
    #     "q_proj",
    #     "v_proj",
    # ],
    lora_target_modules: List[str] = ["query_key_value"],
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter(prompt_template_name)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    # model = LlamaForCausalLM.from_pretrained(
    model = BloomForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    # model = get_peft_model(model, config)
    model_config = model.config.to_dict() if hasattr(model.config, "to_dict") else model.config
    config.base_model_name_or_path = model.__dict__.get("name_or_path", None)
    model = PeftModelForCausalLM(model, config)

    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if val_set_size > 0 else None,
            save_steps=400,  # 200,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


if __name__ == "__main__":
    fire.Fire(train)
