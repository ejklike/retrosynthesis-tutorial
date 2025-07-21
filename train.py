from transformers import (
    EncoderDecoderModel,
    Trainer, TrainingArguments,
    BertConfig, EncoderDecoderConfig, DataCollatorForSeq2Seq
)

from datautil import (
    get_train_val_dataset, tokenize_function, train_tokenizer)


if __name__ == "__main__":
    # Tokenizer 학습
    tokenizer = train_tokenizer([
        "./data/src-train.txt",
        "./data/tgt-train.txt"
    ])

    # Dataset 구성
    raw_datasets = get_train_val_dataset()
    tokenized_datasets = raw_datasets.map(
        lambda x: tokenize_function(x, tokenizer), batched=True)

    # # input: input_ids
    # # output: labels
    # input_smiles = tokenized_datasets['train']['src'][0].replace(' ', '')
    # tokens = tokenizer.convert_ids_to_tokens(tokenized_datasets['train']['input_ids'][0])
    # input_smiles_reconstructed = ''.join(tokens).replace('[PAD]', '')
    # assert input_smiles == input_smiles_reconstructed

    # output_smiles = tokenized_datasets['train']['tgt'][0].replace(' ', '')
    # tokens = tokenizer.convert_ids_to_tokens(tokenized_datasets['train']['labels'][0])
    # output_smiles_reconstructed = ''.join(tokens).replace('[PAD]', '')
    # assert output_smiles == output_smiles_reconstructed

    # 5. Config 정의 및 모델 생성
    config = EncoderDecoderConfig.from_encoder_decoder_configs(
        BertConfig(
            vocab_size=tokenizer.vocab_size,
            bos_token_id=tokenizer.convert_tokens_to_ids("[BOS]"),
            eos_token_id=tokenizer.convert_tokens_to_ids("[EOS]"),
            pad_token_id=tokenizer.convert_tokens_to_ids("[PAD]"),
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=256,
        ),
        BertConfig(
            vocab_size=tokenizer.vocab_size,
            bos_token_id=tokenizer.convert_tokens_to_ids("[BOS]"),
            eos_token_id=tokenizer.convert_tokens_to_ids("[EOS]"),
            pad_token_id=tokenizer.convert_tokens_to_ids("[PAD]"),
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=256,
        )
    )
    model = EncoderDecoderModel(config=config)
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    # 6. TrainingArguments 설정
    training_args = TrainingArguments(
        output_dir="./checkpoints",
        learning_rate=5e-4, #5e-4
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1000,
        weight_decay=0.01,
        save_strategy="epoch",
        eval_strategy="epoch",        # 평가 주기 설정 필수
        load_best_model_at_end=True,        # 가장 좋은 모델 로드
        metric_for_best_model="eval_loss",  # 기준 metric 설정
        greater_is_better=False,             # 낮은 loss가 더 좋으므로 False
        report_to="tensorboard",  # TensorBoard에 로그 기록
        logging_dir="./logs",
        logging_strategy="epoch"
    )

    # 7. Trainer 정의
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )

    # 8. 학습 시작
    trainer.train()


