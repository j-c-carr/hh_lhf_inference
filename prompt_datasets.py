"""
author: @j-c-carr
python script to retrieve prompt datasets
"""
from typing import List, Optional
from datasets import load_dataset, concatenate_datasets


def extract_hh_prompt_from_sample(prompt_and_response: str) -> str:
    """Extract the prompt from a prompt-response pair."""
    search_term = '\n\nAssistant:'
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[:search_term_idx + len(search_term)]


def get_hh_prompts(data_dirs: List[str], split: str, cache_dir: str = None) -> List[str]:
    """Load the prompts from the Anthropic Helpful-Harmless dataset.

       Prompts are be structured as follows:
         \n\nHuman: <prompt>\n\nAssistant:
       Multiple turns are allowed, but the prompt should always start with \n\nHuman: and end with \n\nAssistant:.
    """

    assert len(data_dirs) > 0, 'Must provide at least one data directory'

    print(f'Loading HH ({split} split) from {data_dirs}...')
    datasets = []
    for data_dir in data_dirs:
        dataset = load_dataset('Anthropic/hh-rlhf', split=split, data_dir=data_dir, cache_dir=cache_dir)
        datasets.append(dataset)

    dataset = concatenate_datasets(datasets)
    print('done')

    def extract_hh_prompts(samples):
        """Helper function to get the prompts from the HH dataset"""
        prompt = [extract_hh_prompt_from_sample(chosen_str) for chosen_str in samples['chosen']]
        return {'prompt': prompt}

    # Preprocess the dataset to prepare for TRL training
    original_columns = dataset.column_names
    dataset = dataset.map(extract_hh_prompts, batched=True, remove_columns=original_columns)

    return dataset['prompt']


def get_prompts(dset_name: str, split='train', cache_dir=None, num_samples: Optional[int] = None) -> List[str]:
    """Extracts prompts from the given dataset.
    If :num_samples: is supplied, returns the first :num_samples: samples from the dataset."""
    assert dset_name in ["rtp", "hh",  "hh-helpful-only",  "hh-harmless-only", "fairprism", "xstest"], \
        f'{dset_name} must be in ["rtp", "hh",  "hh-helpful-only", "hh-harmless-only", "fairprism", "xstest"]'

    if dset_name == "rtp":
        print(f'Loading RealToxicityPrompts dataset from Huggingface...')
        dataset = load_dataset('allenai/real-toxicity-prompts', split=split, cache_dir=cache_dir)
        prompts = [prompt['text'] for prompt in dataset['prompt']]

    elif dset_name == "fairprism":
        # Assumes that fairprism_aggregated.csv is in :cache_dir:
        import pandas as pd
        prompts = pd.read_csv(f'{cache_dir}/fairprism_aggregated.csv')['Human Input'].tolist()
        prompts = list(set(prompts))    # Remove duplicate prompts

    elif dset_name == "xstest":
        # Assumes that the xstest prompts are in :cache_dir:
        import pandas as pd
        prompts = pd.read_csv(f'{cache_dir}/xstest_v2_prompts.csv')['prompt'].tolist()

    elif dset_name == "hh":
        data_dirs = ["helpful-base", "harmless-base", "helpful-online", "helpful-rejection-sampled"]
        prompts = get_hh_prompts(data_dirs, split=split, cache_dir=cache_dir)

    elif dset_name == "hh-helpful-only":
        data_dirs = ["helpful-online", "helpful-base", "helpful-rejection-sampled"]
        prompts = get_hh_prompts(data_dirs, split=split, cache_dir=cache_dir)

    elif dset_name == "hh-harmless-only":
        data_dirs = ["harmless-base"]
        prompts = get_hh_prompts(data_dirs, split=split, cache_dir=cache_dir)

    else:
        raise ValueError

    return prompts[:num_samples]


def add_instruction_format(prompts: List[str], dset_name=None) -> List[str]:
    """For the fine-tuned models, format the prompts as they are formatted in the Anthropic HH dataset."""
    if dset_name in ['fairprism', 'xstest']:
        return [f'\n\nHuman: {prompt}\n\nAssistant:' for prompt in prompts]
    else:
        return [f'\n\nHuman: Hi! Could you help me finish a sentence? The sentence is: {prompt}\n\nAssistant:'
                for prompt in prompts]
