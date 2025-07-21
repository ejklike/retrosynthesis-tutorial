import torch
from transformers import EncoderDecoderModel, PreTrainedTokenizerFast
from tqdm import tqdm

from datautil import get_test_dataset, tokenize_function
from rdkitutil import match_smiles


def topk_accuracy(dataset, model, tokenizer, batch_size=8):

    def decode(ids):
        return tokenizer.decode(ids, skip_special_tokens=True)

    topk_accuracies = [0.0] * 10
    total = len(dataset)

    for i in tqdm(range(0, total, batch_size), desc="Evaluating"):

        batch = dataset[i:i+batch_size]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        input_ids = torch.tensor(batch['input_ids']).to(device)
        attention_mask = torch.tensor(batch['attention_mask']).to(device)
        references = batch['tgt']
        actual_batch_size = len(references)  # 마지막 배치에서 중요

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=128,
                num_beams=10,               # 빔 서치 크기 10
                num_return_sequences=10,    # 최대 10개 후보 생성
                early_stopping=True,
                no_repeat_ngram_size=0
            )

        # outputs shape: (batch_size * 10, seq_len)
        # 10 후보씩 묶어서 처리
        outputs = outputs.view(actual_batch_size, 10, -1)

        for j in range(actual_batch_size):
            reference = references[j]
            candidates = [decode(outputs[j, k]) for k in range(10)]
            comparison_results = [match_smiles(reference, cand) for cand in candidates]
            # #############
            # print(f"Reference: {reference.replace(' ', '')}")
            # candidates = [cand.replace(' ', '') for cand in candidates]
            # print("Candidates:")
            # for k, cand in enumerate(candidates):
            #     print(f"  {k+1}: {cand}")
            # print(comparison_results)
            # #############
            # Top-k 정확도 갱신
            for k in range(10):
                if any(comparison_results[:k+1]):
                    # topk_accuracies[k] += 1
                    for m in range(k, 10):
                        topk_accuracies[m] += 1
                    break

    for k in range(10):
        topk_accuracies[k] = topk_accuracies[k] / total * 100
    return topk_accuracies


if __name__ == "__main__":
    
    # 체크포인트 로딩
    checkpoint = "./checkpoints/checkpoint-339021"  # 학습한 체크포인트 경로    
    model = EncoderDecoderModel.from_pretrained(checkpoint)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(checkpoint)

    # 모델을 GPU로 이동
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Dataset 구성
    raw_datasets = get_test_dataset()
    tokenized_datasets = raw_datasets.map(
        lambda x: tokenize_function(x, tokenizer), batched=True)
    
    # top-k 정확도 계산
    print("Calculating top-k accuracy...")
    topk_accuracies = topk_accuracy(tokenized_datasets['test'], 
                                    model, 
                                    tokenizer, 
                                    batch_size=8)

    # 결과 출력
    for k in range(10):
        print(f"Top-{k+1} accuracy: {topk_accuracies[k]:.2f}%")
