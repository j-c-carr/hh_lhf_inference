"""
Main script for generating model outputs on a dataset of test prompts
"""
from tqdm import tqdm
import torch
import transformers
import pandas as pd
from torch.utils.data import DataLoader

from inference_datasets import get_prompts, add_instruction_format

from toxicity_classification import classify_outputs

import gc
gc.collect()
torch.cuda.empty_cache()

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
        outputs.extend([batch_outputs[i, inputs['input_ids'].shape[1]:] for i in range(len(batch_outputs))])

    print('Done.')

    print('Decoding outputs...')
    for i in range(len(outputs)):
        outputs[i] = tokenizer.decode(outputs[i], skip_special_tokens=True)
    print("Len prompts: ", prompts)
    print("Len outputs: ", len(outputs))

    return prompts, outputs 


if __name__ == '__main__':

    # :cache_dir: is the folder containing the dataset.
    # For rtp and hh datasets, set this equal to the hugging face cache folder, e.g.'/network/scratch/j/jonathan.colaco-carr'
    # For FairPrism, or XSTest set :cache_dir: equal to the folder containing the dataset csv file
    dset_name = 'xstest'
    split = 'train' # must be "train" for rtp
    cache_dir = '/network/scratch/j/jonathan.colaco-carr/hh_fruits/data/xstest'
    #cache_dir = None

    classify_toxicity = False

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_samples = None # If None, use the full dataset
    batch_size = 32
    num_return_sequences = 3
    generator_kwargs = {"max_new_tokens": 50,
                        "do_sample": True,
                        "num_return_sequences": num_return_sequences,
                        "top_k": 50,
                        "top_p": 0.95}


    ###############################################
    # Add models for inference here:
    gpt_checkpoint_dir = "/network/scratch/j/jonathan.colaco-carr/hh_fruits/checkpoints/v1_checkpoints/gpt2-large"
    pythia_checkpoint_dir = "/network/scratch/j/jonathan.colaco-carr/hh_fruits/checkpoints/v1_checkpoints/pythia28"

    # keys are the model name, values are the path to the model's weights
    gpt_models = {#'base_lm': None,
              'help_only': f'{gpt_checkpoint_dir}/helpful_only/dpo_gpt2l_helpful_longer_2024-04-13_step-200000_policy.pt',
              'hh_filtered': f'{gpt_checkpoint_dir}/all_filtered/gpt2l_dpo_filtered_longer_2024-04-14_step-280000_policy.pt',
              'hh_harmless': f'{gpt_checkpoint_dir}/hh_harmless_harmless/gpt2l_dpo_harmless_harmless_Jun13.pt',
              'hh_full': f'{gpt_checkpoint_dir}/hh_full/dpo_gpt2l_paper_params_longer_2024-04-13_step-240000_policy.pt'}

    pythia_models = {#'base_lm': None,
              'hh_harmless': f'{pythia_checkpoint_dir}/hh_harmless_harmless/hh_dpo_pythia28_harmless_harmless_Jun14_1epoch.pt',
              'help_only': f'{pythia_checkpoint_dir}/helpful_only/dpo_pythia28_helpful_only_2024_04_16_step-160000.pt',
              'hh_filtered': f'{pythia_checkpoint_dir}/all_filtered/dpo_pythia28_filtered_2024-04-16_step-160000_policy.pt',
              'hh_full': f'{pythia_checkpoint_dir}/hh_full/dpo_pythia28_hh_full_1_epoch.pt'}


    base_model_name = 'EleutherAI/pythia-2.8b' # 'gpt2-large' # use 'EleutherAI/pythia-2.8b' for Pythia models
    models = pythia_models             # use pythia_models for Pythia models
    ###############################################

    # Load the prompts
    prompts = get_prompts(dset_name, split=split, num_samples=num_samples, cache_dir=cache_dir)
    print("Len XSTest Prompts: ", len(prompts))

    # Load custom prompts (for XSTest only)
    if dset_name == 'xstest':
        from xs_custom import disc_prompts, contrast_disc_prompts
        prompts.extend(disc_prompts)
        prompts.extend(contrast_disc_prompts)

    prompt_dataloader = DataLoader(prompts, batch_size=batch_size, shuffle=False)  # DO NOT SHUFFLE!

    # Load the base model
    tokenizer, model = load_tokenizer_and_model(base_model_name, return_tokenizer=True, device=device)

    # Generate outputs for each model
    outputs = {}
    for model_name, model_checkpoint in models.items():
        print('='*80)
        print('Model: ', model_name)

        # Load the model weights
        if model_checkpoint is not None:
            print(f'Loading model checkpoint from {model_checkpoint}...')
            # Convert the state dictionary to half precision
            state_dict = torch.load(model_checkpoint)['state']
            #for key in tqdm(state_dict.keys()):
            #    state_dict[key] = state_dict[key].half()

            model.load_state_dict(state_dict)
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
                                                    **generator_kwargs)


        outputs[f'{model_name}_generations'] = model_generations

        if classify_toxicity:
            toxicity_scores = classify_outputs(model_generations,batch_size=batch_size)
            outputs[f'{model_name}_toxicity'] = toxicity_scores 

    prompts = [prompt for prompt in prompts for _ in range(num_return_sequences)]

    # Save the prompts and outputs
    outputs['prompts'] = prompts
    pd.DataFrame(outputs).to_csv(f'out/{dset_name}_eval/{base_model_name.replace("/","-")}_{dset_name}_{num_return_sequences}_sequences.csv')
