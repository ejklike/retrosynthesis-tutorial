# Retrosynthesis Tutorial

Seq2seq modeing using `transformers`

## Dataset

USPTO-50k (located in `./data/`)

- Source (`src`): product
- Target (`tgt`): reactants
- split into `train`/`val`/`test` sets

## Train Model

You can train a transformer model for retrosynthesis.

```
python train.py
```

You can watch the log using `tensorboard`.

```
tensorboard --logdir=./logs --port xxxx --host 0.0.0.0
```


## Test Model

You can compute the top-k accuracy scores.

```
python evaluate.py
```

