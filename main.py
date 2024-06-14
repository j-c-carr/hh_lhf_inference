"""
Main script for generating model outputs on a dataset of test prompts
Author: @j-c-carr
"""
import os
from tqdm import tqdm
import torch
import transformers
import pandas as pd
from torch.utils.data import DataLoader
from peft import AutoPeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments)
from huggingface_hub import login

from prompt_datasets import get_prompts, add_instruction_format


def load_tokenizer_and_model(name_or_path, model_checkpoint=None, return_tokenizer=False, device='cpu'):
    """Load a model and (optionally) a tokenizer for inference"""
    assert name_or_path in ['gpt2-large', 'EleutherAI/pythia-2.8b'], \
        "name_or_path must be in ['gpt2-large', 'EleutherAI/pythia-2.8b']"

    model = transformers.AutoModelForCausalLM.from_pretrained(name_or_path)

    if model_checkpoint is not None:
        print(f'Loading model checkpoint from {model_checkpoint}...')
        model.load_state_dict(torch.load(model_checkpoint)['state'])
        print('Done.')

    else:
        print(f'No model checkpoint specified. Loading default {name_or_path} model.')

    device = torch.device(device)
    model.to(device)

    if return_tokenizer:
        print('Loading tokenizer...')
        tokenizer = transformers.AutoTokenizer.from_pretrained(name_or_path, padding_side='left')
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        return tokenizer, model

    return model


@torch.no_grad()
def inference_loop(model, tokenizer, prompt_dataloader, device='cpu', instruction_format=False, **generate_kwargs):
    """Use model and tokenizer to tokenizer"""
    model.eval()

    prompts = []
    outputs = []

    print('Generating outputs...')
    for batch_prompts in tqdm(prompt_dataloader):
        prompts.extend(batch_prompts)

        # Add "Human: ... Assistant: ..." for models fine-tuned on Helpful-Harmless dataset
        if instruction_format:
            batch_prompts = add_instruction_format(batch_prompts, dset_name='xstest')

        inputs = tokenizer(batch_prompts, add_special_tokens=False, padding=True, return_tensors='pt').to(device)
        batch_outputs = model.generate(**inputs, **generate_kwargs)

        outputs.extend([batch_outputs[i, inputs['input_ids'].shape[1]:] for i in range(len(batch_prompts))])
    print('Done.')

    # Todo: decode outputs
    print('Decoding outputs...')
    for i in range(len(outputs)):
        outputs[i] = tokenizer.decode(outputs[i], skip_special_tokens=True)
    print('Done.')

    return prompts, outputs 


def big_model_inference():

    # :cache_dir: is the folder containing the dataset.
    # For rtp and hh datasets, set this equal to the hugging face cache folder, e.g.'/network/scratch/j/jonathan.colaco-carr'
    # For FairPrism, or XSTest set :cache_dir: equal to the folder containing the dataset csv file
    dset_name = 'xstest'
    split = 'test'
    cache_dir = '/network/scratch/j/jonathan.colaco-carr/hh_fruits/data/xstest'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_samples = None  # if None, the full prompt dataset is used
    batch_size = 32
    max_new_tokens = 80

    base_model_name = "meta-llama/Meta-Llama-3-8B" #"mistralai/Mistral-7B-v0.1" 

    ###############################################
    # Add models for inference here:

    # Mistral models
    #models = {'help_only_sft': "/network/scratch/j/jonathan.colaco-carr/logs/trl_test/4804235/final_checkpoint",
    #          'help_only_dpo': "/network/scratch/j/jonathan.colaco-carr/logs/trl_test/4804526/final_checkpoint",
    #          'hh_full_sft': "/network/scratch/j/jonathan.colaco-carr/logs/trl_test/4793010/final_checkpoint",
    #          'hh_full_dpo': "/network/scratch/j/jonathan.colaco-carr/logs/trl_test/4796852/final_checkpoint"}

    # Llama models
    models = {'help_only_sft': "/network/scratch/j/jonathan.colaco-carr/logs/trl_test/4804533/final_checkpoint",
              'help_only_dpo': "/network/scratch/j/jonathan.colaco-carr/logs/trl_test/4807943/final_checkpoint",
              'hh_full_sft': "/network/scratch/j/jonathan.colaco-carr/logs/trl_test/4798565/final_checkpoint",
              'hh_full_dpo': "/network/scratch/j/jonathan.colaco-carr/logs/trl_test/4802734/final_checkpoint"}
    ###############################################

    # Load the prompts
    prompts = get_prompts(dset_name, split=split, num_samples=num_samples, cache_dir=cache_dir)
    prompt_dataloader = DataLoader(prompts, batch_size=batch_size, shuffle=False)  # DO NOT SHUFFLE!

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token

    # Generate outputs for each model
    outputs = {}
    for model_name, model_checkpoint in models.items():

        print('='*80)
        print(f'Loading {model_name} from {model_checkpoint}...')
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_checkpoint, # directory of saved model
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            #load_in_4bit=True,
            is_trainable=False,
        )
        model.to(device)
        print('Done.')

        # Add instruction format if necessary
        if (model_checkpoint is not None) and ('hh' not in dset_name):
            instruction_format = True
        else:
            instruction_format = False

        # Generate model outputs
        prompts, model_generations = inference_loop(model, tokenizer,
                                                    prompt_dataloader,
                                                    instruction_format=instruction_format,
                                                    device=device,
                                                    max_new_tokens=max_new_tokens)

        print(model_generations)

        outputs[f'{model_name}_generations'] = model_generations

    # Save the prompts and outputs
    outputs['prompts'] = prompts
    pd.DataFrame(outputs).to_csv(f'out/{base_model_name.split("/")[-1]}_{dset_name}_prompts_{max_new_tokens}_tokens.csv')

if __name__ == '__main__':

    # HF_TOKEN environment variable must be set to a hugging face access token
    login(token=os.environ["HF_TOKEN"])

    big_model_inference()

