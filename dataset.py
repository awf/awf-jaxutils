# Adapted from https://github.com/vpj/jax_transformer

from pathlib import Path
from labml.utils.download import download_file

import jax
import jax.numpy as jnp


class TinyShakespeare:
    """
    ## Tiny Shakespeare dataset
    """

    def __init__(self, rnd_key: jax.random.PRNGKey, seq_len: int, batch_size: int):
        """
        * `rnd_key` is the PRNG state
        * `seq_len` is the sequence length of a sample
        * `batch_size` is the batch size
        """
        print("Dataset init")
        self.batch_size = batch_size
        # PRNG key for shuffling the samples
        _, self.rnd_key = jax.random.split(rnd_key)

        # Local path of the text file
        path = Path("/tmp/tiny_shakespeare.txt")
        # Download if it doesn't exist
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        if not path.exists():
            download_file(url, path)

        # Read the file
        with open(str(path), "r") as f:
            self.text = f.read()

        # Get the characters/tokens
        tokens = sorted(list(set(self.text)))

        # Number of tokens
        self.n_tokens = len(tokens)
        # Map tokens to ids
        self.stoi = {t: i for i, t in enumerate(tokens)}
        # Id to token/character
        self.itos = tokens

        # As a list of ids
        data = jnp.array([self.stoi[s] for s in list(self.text)])
        # Number of batches
        self.n_batches = len(data) // (seq_len * batch_size)
        # Truncate
        data = data[: self.n_batches * seq_len * batch_size]
        # Reshape into a samples (better to use random offsets, but lets ignore that here)
        self.data = data.reshape((-1, seq_len))
        # List of sample indexes
        self.idx = jnp.arange(len(self.data))
        print("Dataset outit")

    def __iter__(self):
        """
        Setup for iteration
        """
        # Iteration step
        self._iter_idx = 0
        # Split PRNG key
        self.rnd_key, rnd_key = jax.random.split(self.rnd_key)
        # Shuffle sample indexes
        self.idx = jax.random.permutation(rnd_key, self.idx)

        #
        return self

    def __len__(self):
        """
        Number of batches
        """
        return self.n_batches

    def __next__(self):
        """
        Get next batch
        """

        # Stop iteration after iterating through all batches
        if self._iter_idx >= self.n_batches:
            raise StopIteration()

        # Sample indexes for the batch
        idx = self.idx[
            self._iter_idx * self.batch_size : (self._iter_idx + 1) * self.batch_size
        ]
        # Increment iteration step
        self._iter_idx += 1

        # Return samples
        return self.data[idx]
