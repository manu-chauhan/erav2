import subprocess
import time

import utilities
from src import HindiTokenizer, BasicTokenizer
from src.HindiTokenizer import SIMPLE_HINDI_PATTERN

if __name__ == "__main__":
    BATCH_SIZE = 30_000
    NUMBER_OF_BATCHES = 50  # None means read all batches
    tokenizer = HindiTokenizer(pattern=SIMPLE_HINDI_PATTERN)
    all_files = utilities.get_all_text_dataset("./dataset")

    # NOTE: first pull all LFS files; src: https://stackoverflow.com/a/72830455/3903762
    process = subprocess.Popen(["git", "lfs", "fetch", "--all"], stdout=subprocess.PIPE)
    output = process.communicate()[0]
    print(output)
    # actually download original file contents, previous command only fetches metadata
    process = subprocess.Popen(["git", "lfs", "checkout"], stdout=subprocess.PIPE)
    output = process.communicate()[0]
    print(output)

    """returns a generator and not a list"""

    # TODO joblib's Parallel should be used to use all CPU cores and update shared data structure to build vocab : to reduce time

    result = utilities.read_from_all_files(all_files, batch_size=BATCH_SIZE, batch_num=NUMBER_OF_BATCHES)
    initial_vocab_size = 400  # initial vocab size to start with, will include 0-255 ASCII chars and most frequent Hindi chars
    vocab_increase_size = 100  # increase vocab size by this much for every batch, will reuse same tokenizer object and vocab

    total_raw_text_len = 0
    start = time.perf_counter()
    # run Tokenizer "train" on each batch ... using same Tokenizer object AND vocab
    for batch_idx, data_batch in enumerate(result):
        print(f"\n\nBatch number=====================================================: {batch_idx}\n")

        batch_text = "".join(data_batch)  # need to join to get single str as batch is list of lines of text

        total_raw_text_len += len(batch_text)

        # as using same tokenizer and same vocab so increasing VOCAB size by `initial_vocab_size` 1st time,
        # initial_vocab_size + 256 is passed only for first iteration,
        # for every subsequent batch iteration increase vocab by vocab_increase_size
        # so first time 200 + 256 = 456, next time 100 (vocab_increase_size) + 456
        if batch_idx == 0:
            tokenizer.train(text=batch_text, verbose=True,
                            vocab_size=initial_vocab_size + 256)  # 256 is for 0-255 ASCII chars AND for consistency
        else:
            tokenizer.train(text=batch_text, verbose=True, vocab_size=vocab_increase_size + 256)

    end = time.perf_counter()
    print(f"\n==============\n\nTime taken for running BPE on entire dataset : {(end - start)} seconds")

    # save the tokenizer object
    tokenizer.save(file_prefix="hindi-30k_batchsize-50_batches-400_initial_vocab-100_next_batches")

    print(f"Total len of text in Hindi from entire dataset: {total_raw_text_len}")

    result = utilities.read_from_all_files(all_files, batch_size=BATCH_SIZE,
                                           batch_num=NUMBER_OF_BATCHES)  # re-reading to encode

    total_encoded_len = 0

    print("\nRunning tokenizer for encoding raw data...\n=================")
    for batch_idx, data_batch in enumerate(result):
        batch_text = "".join(data_batch)
        encoded = tokenizer.encode(text=batch_text)
        print(f"==============\nEncoded batch: {batch_idx + 1}")
        total_encoded_len += len(encoded)

    print(f"Encoded total len: {total_encoded_len}")

    print(f"Ratio of raw data compressed: {total_raw_text_len / total_encoded_len}")
    # print(data_batch)
    # print(len(data_batch))
    # break

    # print(all_files)
    # for file in all_files:
    #     chunk = read_in_chunks(file)
    #     for t in chunk:
    #         print(t)
    #         if len(t) > 2:
    #             tokenizer.train(text=t, verbose=True, vocab_size=300)
    # dataset = get_data_batch(all_files)
    # print(dataset)
    # print(list(get_data_batch(all_files=dataset)))
