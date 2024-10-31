import torch
from torch.utils.data import Dataset, DataLoader

from .utils import read_inputs, count_letters
import os

class CharCountDataset(Dataset):
    def __init__(self, texts, vocab=None, max_len=20):
        """
        Args:
            texts (list of str): Input texts to be tokenized at character level.
            vocab (dict, optional): Character vocabulary mapping to indices. If None, it will be built from data.
            max_len (int, optional): Maximum length of the sequence for padding/truncation.
        """
        self.texts = texts
        self.max_len = max_len
        
        # Build character vocabulary if not provided
        if vocab is None:
            self.vocab = self.build_vocab(self.texts)
        else:
            self.vocab = vocab
            
        self.vocab_size = len(self.vocab)
    
    def build_vocab(self, texts):
        # Create vocabulary from the characters in texts
        vocabs = [chr(ord('a') + i) for i in range(0, 26)] + [' ']
        vocab = {char: idx for idx, char in enumerate(vocabs)}
        return vocab
    
    def char_to_index(self, char):
        # Convert character to index using vocabulary
        return self.vocab.get(char, self.vocab.get('<UNK>'))
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        # Tokenize text into character indices
        input_seq = [self.char_to_index(char) for char in text]
        
        # Get the cumulative character counts as targets
        target_counts = count_letters(text)

        # Convert lists to tensors
        input_tensor = torch.tensor(input_seq, dtype=torch.long)
        target_tensor = torch.tensor(target_counts, dtype=torch.float32)
        
        return input_tensor, target_tensor

    @staticmethod
    def get_dataloader(texts, batch_size=32, max_len=20, vocab=None):
        dataset = CharCountDataset(texts, vocab=vocab, max_len=max_len)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloader

if __name__ == "__main__":
    texts = read_inputs(
        os.path.join("data", "train.txt")
    )

    # Create DataLoader
    dataloader = CharCountDataset.get_dataloader(texts, batch_size=20)

    # Fetch a batch
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}")
        print("Inputs:", inputs)
        print("Targets:", targets)
