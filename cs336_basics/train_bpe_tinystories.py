from pathlib import Path
import os
import pickle
import time
import regex as re
from collections import defaultdict

def gen_vocab_merges(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    dump_path,
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    start_time = time.time()
    vocab, merges = run_train_bpe(
    input_path=input_path,
    vocab_size=vocab_size,
    special_tokens=special_tokens,
)   
    finish_time = time.time()
    print('用时: ', finish_time - start_time)

    max_token_length = -1
    for v in vocab.values():
        if len(v) >= max_token_length:
            max_token_length = len(v)
    longest_tokens = [token for token in vocab.values() if len(token) == max_token_length]
    print('max_token_length: ', max_token_length)
    print('longest_tokens: ', longest_tokens)
    tinystories_vocab_path = dump_path[0]
    tinystories_merges_path = dump_path[1]

    # 创建目录（若不存在）
    Path(tinystories_vocab_path).parent.mkdir(parents=True, exist_ok=True)
    Path(tinystories_merges_path).parent.mkdir(parents=True, exist_ok=True)
    # 序列化保存
    with open(tinystories_vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    with open(tinystories_merges_path, "wb") as f:
        pickle.dump(merges, f)
    print(f"已保存 vocab -> {tinystories_vocab_path}, merges -> {tinystories_merges_path} ")
    return (vocab, merges)

def pre_tokenization(chunk: bytes, chunk_idx: int, COMPILE_PAT, special_tokens,Queue):
    # print('pre_tokenization start, chunk idx: ', chunk_idx)
    pre_tokens = defaultdict(int)
    a = chunk.decode()
    texts_no_special = re.split("|".join(map(re.escape, special_tokens)), a)
    for para in texts_no_special:
        results = re.finditer(COMPILE_PAT, para)
        for i in results:
            word = i.group() # word has form 'str' 
            word = tuple(word.encode()) # str->bytes用encode, bytes->str是decode
            byte_list = list()
            for i in word:
                byte_list.append(bytes([i]))
            word = tuple(byte_list)
            pre_tokens[word] += 1
    Queue.put((chunk_idx, pre_tokens))
    # print(f'chunk {chunk_idx} put to queue')

def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    start_time = time.time()
    # from ..cs336_basics.pretokenization_example import find_chunk_boundaries
    def find_chunk_boundaries(
        file,
        desired_num_chunks: int,
        split_special_token: bytes,
    ) -> list[int]:
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """
        assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        chunk_size = file_size // desired_num_chunks
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size
        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk
                # 每次只处理一小部分内容。这样可以有效地控制内存使用，并避免一次性加载整个文件到内存中。
                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break
                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size
        return sorted(set(chunk_boundaries))
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    COMPILE_PAT = re.compile(PAT)
    # pretokenization
    import multiprocessing
    Queue = multiprocessing.Queue()
    pre_tokens = defaultdict(int)

    chunk_idx = 0
    proc_list = []
    num_processes = 10
    with open(input_path, 'rb') as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        for start, end in zip(boundaries[:-1], boundaries[1:]): #  “从列表开头取到倒数第二个元素”（排除最后一个元素）
            f.seek(start)
            chunk = f.read(end - start)
            p = multiprocessing.Process(target= pre_tokenization, args=(chunk, chunk_idx, COMPILE_PAT, special_tokens, Queue))
            p.start()
            proc_list.append(p)
            chunk_idx += 1
    c = 0
    while c < len(proc_list):
        chunk_idx, pre_token_return = Queue.get()
        # print(f'No. {chunk_idx} value returned.')
        c += 1
        for token in pre_token_return:
            pre_tokens[token] += pre_token_return[token]
    for p in proc_list:
        p.join()
    # pretokenization complete
    pre_token_time = time.time()
    print('\npre_tokenization time: ', pre_token_time - start_time)
    # init vocab
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(0,256)}
    new_idx = len(vocab)
    special_token_bytes = [token.encode("utf-8") for token in special_tokens]
    for token_bytes in special_token_bytes:
        if token_bytes not in vocab.values():
            vocab[new_idx] = token_bytes
            new_idx += 1

    merges: list[tuple[bytes, bytes]] = []
    pair_cnt = defaultdict(int)
    for token in pre_tokens:
        j = 0
        while j < len(token) - 1: 
            pair = (token[j], token[j+1])
            pair_cnt[pair] += pre_tokens[token]
            j += 1
    while len(vocab) < vocab_size:
        if len(pair_cnt.values()) == 0:  
            #print('pair count break')
            break # every token is merged into single byte sequence. Fully tokenized
        max_count = max(pair_cnt.values())
        candidates = [pair for pair in pair_cnt if pair_cnt[pair] == max_count]
        max_pair = max(candidates)
        # print(max_pair)
        byte1, byte2 = max_pair
        # enlarge the vocab and merges
        vocab[new_idx] = byte1 + byte2
        # print(new_idx)
        new_idx += 1
        merges.append(max_pair)
        # merge the pre_tokens
        # pre_tokens cannot be changed during iteration
        token_changes = []
        for token in pre_tokens:
            new_token = []
            j = 0
            changed = False
            while j < len(token):
                if j + 1 < len(token) and (token[j], token[j+1]) == max_pair:
                    new_token.append(vocab[new_idx-1])
                    j += 2
                    changed = True
                else:
                    new_token.append(token[j])
                    j += 1
            if(changed): token_changes.append((token, tuple(new_token), pre_tokens[token]))
        for change in token_changes:
            t_old, t_new, cnt = change
            pre_tokens[t_new] = cnt
            i, j = 0, 0
            while i < len(t_old) - 1:
                broken_pair = (t_old[i], t_old[i+1])
                pair_cnt[broken_pair] -= pre_tokens[t_old]
                i += 1
            while j < len(t_new) - 1:
                add_pair = (t_new[j], t_new[j+1])
                pair_cnt[add_pair] += pre_tokens[t_new]
                j += 1
            del pre_tokens[t_old]
        
    print('loop time: ', time.time() - pre_token_time)
    return (vocab, merges)

if __name__ == '__main__':
    # 绝对导入写法仅适用于vscode内部调试
    # from scalene.scalene_profiler import enable_profiling
    input_path = 'D:/CollegeLife/self_learning/CS336 2025/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt'
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    tinystories_vocab_path = './data/tinystories_vocab_10000.pkl'
    tinystories_merges_path = './data/tinystories_merges_10000.pkl'
    dump_path = [tinystories_vocab_path, tinystories_merges_path]
    gen_vocab_merges(input_path, vocab_size, special_tokens, dump_path)
    with open(tinystories_vocab_path, "rb") as f:
        vocab = pickle.load(f)
    with open(tinystories_merges_path, "rb") as f:
        merges = pickle.load(f)

    # input_path = 'D:/CollegeLife/self_learning/CS336 2025/assignment1-basics/data/owt_train.txt'
    # vocab_size = 32000
    # special_tokens = ["<|endoftext|>"]
    # OpenWebText_vocab_path = './data/OpenWebText_vocab_32000.pkl'
    # OpenWebText_merges_path = './data/OpenWebText_merges_32000.pkl'
    # dump_path = [OpenWebText_vocab_path, OpenWebText_merges_path]
    # gen_vocab_merges(input_path, vocab_size, special_tokens, dump_path)
    # with open(OpenWebText_vocab_path, "rb") as f:
    #     vocab = pickle.load(f)
    # with open(OpenWebText_merges_path, "rb") as f:
    #     merges = pickle.load(f)

    pass
