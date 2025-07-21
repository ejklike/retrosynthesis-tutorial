import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizerFast


# 1. Tokenizer 학습 (BPE 기반)
def train_tokenizer(files, vocab_size=100):
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=vocab_size, 
                         special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"])
    tokenizer.train(files, trainer)
    tokenizer.save("smiles-tokenizer.json")
    return PreTrainedTokenizerFast(tokenizer_file="smiles-tokenizer.json",
                                   bos_token="[BOS]", 
                                   eos_token="[EOS]", 
                                   pad_token="[PAD]", 
                                   unk_token="[UNK]")

# 2. 데이터 불러오기
def load_data(split):
    df_src = pd.read_csv(f"data/src-{split}.txt", header=None, names=["src"])
    df_tgt = pd.read_csv(f"./data/tgt-{split}.txt", header=None, names=["tgt"])
    return Dataset.from_pandas(pd.concat([df_src, df_tgt], axis=1))

# 3. 데이터셋 전체 구성
def get_train_val_dataset():
    return DatasetDict({
        "train": load_data("train"),
        "validation": load_data("val"),
        "test": load_data("test")
    })

def get_test_dataset():
    return DatasetDict({
        "test": load_data("test")
    })

# 4. Tokenizer 적용
def tokenize_function(example, tokenizer):
    model_inputs = tokenizer(
        example["src"], 
        text_target=example["tgt"],
        max_length=128,
        padding="max_length",
        truncation=True,
        return_attention_mask=True)
    return model_inputs