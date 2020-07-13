import tokenizers

DEVICE='cpu'#cuda
MAX_LEN = 128
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 8
EPOCHS = 6
BERT_PATH = "../pretrained_models/roberta-base/"
MODEL_PATH = "pytorch_model.bin"
TRAINING_FILE = "../data/train.csv"
TEST_FILE="../data/test.csv"
TOKENIZER = tokenizers.ByteLevelBPETokenizer(
    vocab_file="../pretrained_models/roberta-base/vocab.json",
    merges_file="../pretrained_models/roberta-base/merges.txt",
    lowercase=True,
    add_prefix_space=True
)
