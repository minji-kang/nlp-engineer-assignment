import numpy as np
import os
import uvicorn, nest_asyncio
import torch

from src.nlp_engineer_assignment import (
  count_letters, print_line, read_inputs, score, train_classifier
)
from src.nlp_engineer_assignment.dataset import CharCountDataset


def train_model():
    cur_dir = os.path.dirname(os.path.abspath(__file__))

    ###
    # Setup
    ###

    # Constructs the vocabulary as described in the assignment
    vocabs = [chr(ord('a') + i) for i in range(0, 26)] + [' ']

    ###
    # Train
    ###

    train_inputs = read_inputs(
        os.path.join(cur_dir, "data", "train.txt")
    )

    model = train_classifier(train_inputs)

    ###
    # Test
    ###

    test_inputs = read_inputs(
        os.path.join(cur_dir, "data", "test.txt")
    )

    model.eval()

    test_dataset = CharCountDataset(texts=test_inputs)

    golds = []
    predictions = []
    for text, gold in test_dataset:
        logits, prediction = model.generate(text)
        predictions.append(prediction.cpu().numpy().flatten())
        golds.append(gold.numpy())

    golds = np.stack(golds)
    predictions = np.stack(predictions)

    # Print the first five inputs, golds, and predictions for analysis
    for i in range(5):
        print(f"Input {i+1}: {test_inputs[i]}")
        print(f"Gold {i+1}: {count_letters(test_inputs[i]).tolist()}")
        print(f"Pred {i+1}: {predictions[i].tolist()}")
        print_line()

    print(f"Test Accuracy: {100.0 * score(golds, predictions):.2f}%")
    print_line()

    torch.save({
            'model_state_dict': model.state_dict(),
            }, 'data/trained_model.ckpt')


if __name__ == "__main__":
    train_model()
    