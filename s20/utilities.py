import fileinput
import json
import os
import pathlib
import sys
from functools import wraps
from typing import List, Union


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def log_to_file(file_name="Default.log"):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Save the current stdout and stderr
            original_stdout = sys.stdout
            original_stderr = sys.stderr

            # Redirect stdout and stderr to the log file
            logger = Logger(file_name)
            sys.stdout = logger
            sys.stderr = logger

            try:
                # Call the original function
                result = func(*args, **kwargs)
                return result
            finally:
                # Reset stdout and stderr
                sys.stdout = original_stdout
                sys.stderr = original_stderr

        return wrapper

    return decorator


def read_from_all_files(all_files_to_read: List[Union[str, pathlib.Path]], batch_size: int = 1000,
                        batch_num: int = None,
                        encoding: str = "utf-8",
                        reading_only_specific_files: List[str] = None) -> List:
    """
    bas basic generator that yields a batch of lines, leverages in-built fileinput for reading all files and using same file object
    :param all_files_to_read: list of file paths, str or Path
    :param batch_size: the number of maximum lines to yield
    :param batch_num: the number of batches to yield and then stop, added later for testing
    :return: List of text lines
    """
    print("\n=========\nReading dataset\n=============")
    counter = 0
    if reading_only_specific_files:
        for idx, f_name in enumerate(all_files_to_read):
            if not all(x in f_name for x in reading_only_specific_files):
                all_files_to_read.pop(idx)

    print(f"\nCount of files to read...{len(all_files_to_read)}")

    with fileinput.input(files=all_files_to_read,
                         encoding=encoding) as f:  # in-built fileinput to read all files, efficient, handles things internally
        if reading_only_specific_files:
            batch = []
        for line in f:
            # print(f"file number: {f.fileno()}")
            # print(f"file-line number: {f.filelineno()}")
            # print(line)
            if line != '\n':
                batch.append(line)
            if len(batch) == batch_size:
                counter += 1
                yield batch
                if batch_num and counter == batch_num:
                    break
                batch = []
        if batch:
            yield batch
        print(f"\nFinal counter value: {counter}")
        print("\n=========\nReading dataset done\n=============")


def read_chunks_from_file(file_path, chunk_size=4 * 1024 * 1024, encoding="utf-8"):
    """
    helper function to yield chunk_size of data read from the file_path given
    """
    file_path = os.path.abspath(file_path)
    with open(file_path, 'r', encoding=encoding) as f:
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
    # first convert json data to text and then process text
    convert_json_data_to_text_and_process_text(dir_path="./web-scrapper",
                                               file_type=".json",
                                               output_file_path="./dataset/combined_from_crawler-json.txt")

    for txt_file in pathlib.Path(path).rglob('*' + file_type):
        files.append(txt_file)
    return files


# def get_data_batch(all_files, chunk_size=100 * 1024 * 1024, formats=".txt"):
#     for file in all_files:
#         yield from read_chunks_from_file(file)


def convert_json_data_to_text_and_process_text(dir_path, file_type=".json", output_file_path="crawler_data.txt"):
    """
    Helper function to convert JSON data to text and then process the text

    """

    with open(output_file_path, "w", encoding="utf-8") as f_out:
        for json_file in pathlib.Path(dir_path).rglob('*' + file_type):
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    f_out.write(" ".join(item["text"]) + "\n")
