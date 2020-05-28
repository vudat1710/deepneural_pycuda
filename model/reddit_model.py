from model.tensor_new import *

__all__ = ['RedditModel']


class RedditModel(Layer):
    def __init__(
            self,
            user_embedding: Embedding,
            sub_embedding: Embedding,
            word_embedding: Embedding,
            hidden_dim,
            output_dim,
            device='cuda',
            p_dropout=0.,
            **kwargs,
    ):
        assert user_embedding is not None, 'user_embedding is required'
        assert sub_embedding is not None, 'sub_embedding is required'
        assert word_embedding is not None, 'word_embedding is required'
        assert user_embedding.embedding_dim == word_embedding.embedding_dim == sub_embedding.embedding_dim, \
            'user_embedding, sub_embedding, word_embedding must have the same dim.'

        self.device = device
        self.user_embedding = user_embedding
        self.sub_embedding = sub_embedding
        self.word_embedding = word_embedding
        input_dim = word_embedding.embedding_dim

        self.lstm = LSTMLayer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            device=device,
        )

        self.h2o = Sequential(layers=[
            # ReLu(),
            # Dropout(p=p_dropout),
            Linear(
                n_inputs=hidden_dim,
                n_outputs=hidden_dim // 2,
                bias=True,
                device=device,
            ),
            Tanh(),
            Dropout(p=p_dropout),
            Linear(
                n_inputs=hidden_dim // 2,
                n_outputs=output_dim,
                bias=True,
                device=device,
            )
        ])

    def forward(self, x):
        users, source_subs, target_subs, contents = x
        inputs = [
            self.user_embedding(users),
            self.sub_embedding(source_subs),
            self.sub_embedding(target_subs),
            *self.word_embedding(contents),
        ]
        hs, _ = self.lstm(inputs=inputs)
        return self.h2o(hs[-1])

    def __call__(self, x):
        return self.forward(x)
