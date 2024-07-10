import os
import pathlib
import time
from typing import List, Union
from src import BasicTokenizer, HindiTokenizer
import fileinput


def read_from_all_files(all_files_to_read: List[Union[str, pathlib.Path]], batch_size: int = 1000) -> List:
    """
    bas basic generator that yields a batch of lines, leverages in-built fileinput for reading all files and using same file object
    :param all_files_to_read: list of file paths, str or Path
    :param batch_size: the number of maximum lines to yield
    :return: List of text lines
    """
    print("\n=========\nReading dataset\n=============")
    with fileinput.input(files=all_files_to_read, encoding="utf-8") as f:
        batch = []
        for line in f:
            # print(f"file number: {f.fileno()}")
            # print(f"file-line number: {f.filelineno()}")
            # print(line)
            if line != '\n':
                batch.append(line)
            if len(batch) == batch_size:
                yield batch
                batch = []


def read_chunks_from_file(file_path, chunk_size=4 * 1024 * 1024):
    """
    helper function to yield chunk_size of data read from the file_path given
    """
    file_path = os.path.abspath(file_path)
    with open(file_path, 'r', encoding="utf-8") as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            yield chunk


def get_all_text_dataset(path: str | pathlib.Path, file_type=".txt") -> List:
    """
    Helper function to get all .txt files' given a path or root directory, uses glob recursively to find the given format files
    :param path: str or Path object, root directory for a dataset
    :param file_type: format of files to get
    :return: list of path of all files of the specified format
    """
    files = []
    for txt_file in pathlib.Path(path).rglob('*' + file_type):
        files.append(txt_file)
    return files


# def get_data_batch(all_files, chunk_size=100 * 1024 * 1024, formats=".txt"):
#     for file in all_files:
#         yield from read_chunks_from_file(file)

if __name__ == "__main__":
    BATCH_SIZE = 1000
    tokenizer =  HindiTokenizer()  # BasicTokenizer()
    all_files = get_all_text_dataset("./dataset")

    """returns a generator and not a list"""

    # TODO joblib's Parallel should be used to use all CPU cores and update shared data structure to reduce time

    result = read_from_all_files(all_files, batch_size=BATCH_SIZE)
    initial_vocab_size = 300
    vocab_increase_size = 50
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
            tokenizer.train(text=batch_text, verbose=True, vocab_size=initial_vocab_size + 256)
        else:
            tokenizer.train(text=batch_text, verbose=True, vocab_size=vocab_increase_size + 256)

    end = time.perf_counter()
    print(f"\n==============\n\nTime taken for running BPE on entire dataset : {(end - start)} seconds")

    tokenizer.save(file_prefix="test-hindi-1")

    print(f"Total len of text in Hindi from entire dataset: {total_raw_text_len}")

    result = read_from_all_files(all_files, batch_size=BATCH_SIZE)

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
