import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from tqdm import tqdm


class Model(nn.Module):
    def __init__(
            self,
            embedding,
            hidden_dim,
            output_dim,
            **kwargs,
    ):
        super(Model, self).__init__()
        if embedding is not None:
            self.embedding = embedding
            self.embedding_dim = embedding.embedding_dim
        else:
            vocab_size = kwargs.get('vocab_size')
            embedding_dim = kwargs.get('embedding_dim')
            assert vocab_size is not None, 'vocab_size is required'
            assert embedding_dim is not None, 'embedding_dim is required'
            self.embedding_dim = embedding_dim
            self.embedding = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=embedding_dim,
            )

        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=hidden_dim,
        )

        self.h2o = nn.Linear(
            in_features=hidden_dim,
            out_features=output_dim,
            bias=True,
        )

    def forward(self, input):
        hs, _ = self.lstm(self.embedding(input))
        output = self.h2o(hs[-1])
        return output

    def __call__(self, input):
        return self.forward(input)
    

if __name__ == '__main__':
    model = Model(
        embedding=None, 
        hidden_dim=300,
        output_dim=10,
        embedding_dim=300,
        vocab_size=10000,
    )

    model = model.cuda()
    x = torch.Tensor([[*range(i, i + 20)] for i in range(20)]).long().cuda()
    target = torch.Tensor(np.random.randint(0, 10, 20)).long().cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=model.parameters(), lr=1e-2)
    for _ in tqdm(range(0, 10)):
        output = model(x)
        loss = criterion(output, target)
        loss.backward()
        optimizer.zero_grad()
        optimizer.step()





