# COMPARISON: the same model in PyTorch and Keras

For the Lakeshore Bike Share manager and the two team members.

## What I observed

1. **Code for training.** The PyTorch loop is two functions in `train.py`, `run_epoch` and `train_model`, about 56 lines (lines 34-61 and 64-91). I had to write the best-weights logic myself with `copy.deepcopy(model.state_dict())`. In Keras the same work is one `compile` call, two callbacks (`EarlyStopping` with `restore_best_weights=True`, and `ModelCheckpoint`) and one `fit` call in `train_keras.py`.
2. **Time.** Over seeds 2026 to 2028, 200 epochs took about 3.5 seconds in PyTorch and about 8.0 seconds in Keras on my laptop, so Keras was about 2.3 times slower (`reports/comparison.csv`).
3. **Accuracy.** PyTorch had the lower error on every seed: mean validation MAE 2.615 against 2.821, and mean test MAE 2.570 against 2.726. The gap is small. It is about the same size as the spread between seeds inside one framework (PyTorch test MAE ranged from 2.467 to 2.661). The frameworks also start from different initial weights, so I cannot say the framework itself causes the gap. Keras seed 2027 had its best epoch at 200, so Keras may need more epochs.
4. **Saving.** `ModelCheckpoint` wrote one file, `reports/best_model_keras.keras` (81 KB), that `keras.models.load_model` reloads without my network class. My PyTorch loop keeps the best weights only in memory. Saving them to disk would need `torch.save`, and loading them back would need the `RentalNet` class.

## Recommendation

I recommend PyTorch. The main reason is the scenario's possible move to Hugging Face: as far as I know, most pretrained models there are published for PyTorch, so staying in PyTorch avoids a second switch later. Our PyTorch version is also faster per run and was at least as accurate in my three seeds, and the loop and its tests already exist. For a small team with a weekly job, one framework that fits both the current model and the later models keeps the job simple.

## What the team gives up

The team gives up Keras's short training call, its ready-made callbacks and its one-file saving. We keep a loop of about 56 lines that someone has to get right, and we must save and reload the best weights by hand.

## What would change my recommendation

If the team decided never to use Hugging Face models, and the weekly job was handed to someone who only knows Keras, Keras would be less code to maintain, and I would recommend it.