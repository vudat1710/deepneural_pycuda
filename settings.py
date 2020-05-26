MAX_LENGTH = 50

# CSV
POST_CROSS_LINKS_FN = 'data/prediction/detailed_data/post_crosslinks_info.tsv'
LABEL_INFO_FN = 'data/prediction/detailed_data/label_info.tsv'
POST_INFO_FN = 'data/prediction/detailed_data/tokenized_posts.tsv'

# DUMP processed csv
POST_CROSS_LINKS_DUMP_FN = 'data/dump/post_crosslinks_info.pkl'
LABEL_INFO_DUMP_FN = 'data/dump/label_info.pkl'
POST_INFO_DUMP_FN = 'data/dump/post_info.pkl'
PROCESSED_DATA = 'data/dump/processed_data.pkl'

# embedding file
GLOVE_EMBEDDING_FN = 'data/prediction/embeddings/glove_word_embeds.txt'
SUB_VECS_FN = 'data/prediction/embeddings/sub_vecs.npy'
SUB_VOCAB_FN = 'data/prediction/embeddings/sub_vecs.vocab'
USER_VECS_FN = 'data/prediction/embeddings/user_vecs.npy'
USER_VOCAB_FN = 'data/prediction/embeddings/user_vecs.vocab'

# train, test, dev dump
TRAIN_DF_DUMP = 'data/dump/train_data.pkl'
TEST_DF_DUMP = 'data/dump/test_data.pkl'
DEV_DF_DUMP = 'data/dump/dev_data.pkl'

# batch_size
TRAIN_BS = 512
TEST_BS = 512
DEV_BS = 512

# DEVICE
DEVICE = 'cuda'

# num epochs
NUM_EPOCHS = 10

# learning rate
LR = 0.001
