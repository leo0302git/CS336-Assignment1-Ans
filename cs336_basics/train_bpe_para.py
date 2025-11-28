from pretokenization_example import find_chunk_boundaries
import multiprocessing
import regex as re
import pickle
from pathlib import Path
import time
import os
import queue

def save_checkpoint(
    checkpoint_path: str,
    vocab: dict,
    merges: list,
    total_pair_count: dict,
    current_iteration: int,
    freq_table: dict
    # chunk_idx : int
):
    """保存中间状态到 checkpoint 文件"""
    checkpoint = {
        "vocab": vocab,
        "merges": merges,
        'total_pair_count': total_pair_count,
        'current_iteration': current_iteration,
        'freq_table': freq_table
        # 'chunk_idx':chunk_idx
    }
    # 创建目录（若不存在）
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    # 序列化保存
    with open(checkpoint_path, "wb") as f:
        pickle.dump(checkpoint, f)
    print(f"已保存 checkpoint 到 {checkpoint_path}, current_iteration: {current_iteration}")

def load_checkpoint(checkpoint_path: str) -> dict:
    """从 checkpoint 文件加载中间状态"""
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint 文件不存在：{checkpoint_path}")
    with open(checkpoint_path, "rb") as f:
        checkpoint = pickle.load(f)
    print(f"已加载 checkpoint 当前轮次：{checkpoint['current_iteration']}")
    return checkpoint

def has_subtuple(main_tuple, sub_tuple):
    cnt = 0
    sub_len = len(sub_tuple)
    main_len = len(main_tuple)
    # 若子元组长度大于主元组，直接返回 False
    if sub_len > main_len:
        return False
    # 遍历主元组中可能的起始位置
    for i in range(main_len - sub_len + 1):
        # 截取主元组中长度为 sub_len 的连续元素，与子元组比较
        if main_tuple[i:i+sub_len] == sub_tuple:
            cnt += 1
    return cnt

def init_byte_pair_count(freq_table_split_num):
    byte_pair_count: dict[tuple, int] = {}
    for word_split_tuple in freq_table_split_num: # 只在pre_tokens内部的某一个元组，而非跨元组进行BPE
        for index1, index2 in zip(word_split_tuple[0:], word_split_tuple[1:]):
            byte_pair_count[(index1, index2)] = byte_pair_count.get((index1, index2), 0) + freq_table_split_num[word_split_tuple]
    return byte_pair_count

def update_byte_pair_count(freq_table_split_num, byte_pair_count, vocab, idx1 = -1, idx2 = -1, new_idx = -1):
    '''只更新byte_pair_count, 不更新freq_table_split_num(留到merge做)'''
    broken_pair = {}
    add_pair = {}
    for tup in freq_table_split_num:
        if has_subtuple(tup, (idx1, idx2)) > 0:
            broken_pair_incre, add_pair_incre = pair_changed(tup, (idx1, idx2), freq_table_split_num[tup], new_idx)
            for pair in broken_pair_incre:
                broken_pair[pair] = broken_pair.get(pair, 0) + broken_pair_incre[pair]
            for pair in add_pair_incre:
                add_pair[pair] = add_pair.get(pair, 0) + add_pair_incre[pair]
    # for tup in freq_table_split_num:
    #     j = 0
    #     # 这个while的作用是获得所有破坏对和新增对（对于多最大对的情况，同时处理）
    #     while j < len(tup) - 1:
    #         if (tup[j], tup[j+1]) == (idx1, idx2):
    #             broken_pair[(idx1, idx2)] = broken_pair.get((idx1, idx2), 0) + freq_table_split_num[tup]
    #             if j + 3 < len(tup) and (tup[j+2], tup[j+3]) == (idx1, idx2):
    #                 # [in][in]
    #                 add_pair[(new_idx, new_idx)] = add_pair.get((new_idx, new_idx), 0) + freq_table_split_num[tup]
    #             if j > 0: 
    #                 front_pair = (tup[j-1], tup[j])
    #                 broken_pair[front_pair] = broken_pair.get(front_pair, 0) + freq_table_split_num[tup]
    #                 front_new_pair = (tup[j-1], new_idx)
    #                 add_pair[front_new_pair] = add_pair.get(front_new_pair, 0) + freq_table_split_num[tup]
    #             if j + 2 < len(tup):
    #                 back_pair = (tup[j+1], tup[j+2])
    #                 broken_pair[back_pair] = broken_pair.get(back_pair, 0) + freq_table_split_num[tup]
    #                 back_new_pair = (new_idx, tup[j+2])
    #                 add_pair[back_new_pair] = add_pair.get(back_new_pair, 0) + freq_table_split_num[tup]
    #         j += 1
    for pair, value in add_pair.items():
        byte_pair_count[pair] = byte_pair_count.get(pair, 0) + value

    for pair, value in broken_pair.items():
        if pair not in byte_pair_count:
            print(pair, '不在byte_pair_count中')
            continue
        byte_pair_count[pair] = byte_pair_count[pair] - value
        try:
            assert byte_pair_count[pair] >= 0
        except Exception:
            print('使得计数值为负的pair ', pair)
            continue
        #print('将索引对 ', pair, '(即', (vocab[pair[0]].decode(), vocab[pair[1]].decode()),')', '的计数减 ', value, ',更新后：', byte_pair_count[pair])
    # if len(broken_pair) == 0 and len(add_pair) == 0: return True
    # else: return False
    pass
    
    return byte_pair_count

def max_tuple(byte_pair_count, merges, vocab):
    max_count = max(byte_pair_count.values()) # 先按count的值，选出最多的元组序列
    candidates = [pair for pair in byte_pair_count if byte_pair_count[pair] == max_count] 
    new_candidates = []
    # 元组比较规则：先比第一个元素，若相同则比第二个元素 最好不要在遍历一个list的同时改变它
    for candidate in candidates:
        if (vocab[candidate[0]], vocab[candidate[1]]) not in merges: 
            new_candidates.append(candidate)
        else: print('merge already have: ', candidate)
    max_tuple = max(new_candidates) # 再在这些元组中选择元组内数字最大的那个
    return max_tuple

def idx2letter_dict(pre_tokens: dict, vocab: dict):
    letters = {}
    for tup in pre_tokens:
        t = tuple()
        for item in tup:
            try:
                t += ((vocab[item]).decode(),)
            except UnicodeDecodeError:
                # print('无法被解码：', (vocab[item]))
                continue
        letters[t] = pre_tokens[tup]
    return letters
def count_pair(tup, tupnum):
    pair = {}
    for index1, index2 in zip(tup[0:], tup[1:]):
        pair[(index1, index2)] = pair.get((index1, index2), 0) + tupnum
    return pair

def merge_tup(tup, idx1, idx2, new_idx):
    j = 0
    new_tup = tuple()
    while j < len(tup): # 仿照课件写法
            if j < len(tup) - 1 and tup[j] == idx1 and tup[j+1] == idx2:
                new_tup += (new_idx,)
                j += 2
            else:
                new_tup += (tup[j],)
                j += 1
    return new_tup

def pair_changed(tup, max_tup, tupnum, new_idx):
    idx1, idx2 = max_tup
    broken_pair = count_pair(tup, tupnum)
    new_tup = merge_tup(tup, idx1, idx2, new_idx)
    add_pair = count_pair(new_tup, tupnum)
    return broken_pair, add_pair
def merge(freq_table_split_num,idx1,idx2,new_idx):
    new_freq_table = {}
    for tup in freq_table_split_num:
        j = 0
        new_tup = merge_tup(tup, idx1, idx2, new_idx)
        new_freq_table[new_tup] = freq_table_split_num[tup]
    return new_freq_table

def pre_tokenize_chunk(chunk: str, chunk_idx: int):
    freq_table = {}
    stories = chunk.split('<|endoftext|>')
    # print('Chunk ', chunk_idx, 'Story num: ', len(stories), end='\n')
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for story in stories:
        words_in_story = re.findall(PAT,story)
        for word in words_in_story:
            if word in freq_table: freq_table[word] += 1
            else: freq_table[word] = 1
        #print(story_split)
    # 将这个chunk中所有story的词都放入freq_table中后，开始对字节对计数
    freq_table_split_letter: dict[tuple, int] = {}
    freq_table_split_num: dict[tuple, int] = {}
    for word in freq_table:
        index_tuple = tuple() # (108, 111, 119)
        letter_tuple = tuple(word)
        for letter in letter_tuple:
            # if ord(letter) > 256:
            #     print('特殊字符 ',ord(letter),bytes(letter.encode()),letter)
            index_tuple += tuple(bytes(letter.encode())) # 添加单个元素进元组需要加逗号。为了正确映射字节到字符，需要加上offset
        freq_table_split_num[index_tuple] = freq_table[word]
        freq_table_split_letter[letter_tuple] = freq_table[word]
    return freq_table_split_num
def check_all_tokenized(freq_table_split_num) -> bool:
    for tup in freq_table_split_num:
        if len(tup) > 1: return False
    return True

def pre(chunk: str, chunk_idx:int, queue):
    try:
        #print(f'No. {chunk_idx} start!')
        freq_table_split_num = pre_tokenize_chunk(chunk, chunk_idx)
        byte_pair_count = init_byte_pair_count(freq_table_split_num)
        queue.put((chunk_idx, freq_table_split_num, byte_pair_count))  # 正常数据
        #print(f'No. {chunk_idx} finished processing')  # 新增：标记子进程完成
    except Exception as e:
        print(f'No. {chunk_idx} failed: {str(e)}')  # 打印错误详情
        queue.put((chunk_idx, None, None))  # 发送错误标记，避免主进程等待

def train_bpe_para(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]: 
    # 约定了类型，当有任何函数分支返回不了规定的类型时就会报错
    '''
    input:
    input_path: str Path to a text file with BPE tokenizer training data.  
    vocab_size: int A positive integer that defines the maximum final vocabulary size (including the initial byte vocabulary, vocabulary items produced from merging, and any special tokens).  
    special_tokens: list[str] A list of strings to add to the vocabulary. These special tokens do not otherwise affect BPE training.  

    return:
    vocab: dict[int, bytes] The tokenizer vocabulary, a mapping from int (token ID in the vocabulary) to bytes (token bytes).  
    merges: list[tuple[bytes, bytes]] A list of BPE merges produced from training. 
    Each list item is a tuple of bytes (<token1>, <token2>), representing that <token1> was merged with <token2>. The merges should be ordered by order of creation.
    '''
    restart = True
    check_period = 500
    start_time = time.time()
    checkpoint_path = './data/bpe_checkpoint.pkl'
    if restart:
        merges: list[tuple[bytes, bytes]] = []
        vocab: dict = {x: bytes([x]) for x in range(256)}  # index -> bytes
        for i, s in enumerate(special_tokens):
            if s not in vocab: vocab[i + 256] = s
        original_vocab_len = len(vocab)
        current_iteration = 0
        byte_pair_count = {}
        total_pair_count: dict[tuple, int] = {}
        freq_table: dict[int, dict] = {}
    else:
        checkpoint = load_checkpoint(checkpoint_path)
        vocab = checkpoint['vocab']
        merges = checkpoint['merges']
        total_pair_count = checkpoint["total_pair_count"]
        current_iteration = checkpoint["current_iteration"]
        original_vocab_len = len(vocab)
        freq_table = checkpoint['freq_table']
    chunk_idx = 0
    num_processes = 2 # downscale
    proc_list = []
    queue = multiprocessing.Queue()
    if restart:
        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
            for start, end in zip(boundaries[:-1], boundaries[1:]): # downscale
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                p = multiprocessing.Process(target=pre, args=(chunk, chunk_idx, queue))
                proc_list.append(p)
                p.start()
                chunk_idx += 1

            c = 0
            while c < len(proc_list):
                chunk_idx, freq_table_split_num, byte_pair_count = queue.get()
                #print(f'No. {chunk_idx} value returned.')
                c += 1
                if len(freq_table_split_num) > 0: freq_table[chunk_idx] = freq_table_split_num
                for pair in byte_pair_count:
                    total_pair_count[pair] = total_pair_count.get(pair, 0) + byte_pair_count[pair]
            for p in proc_list:
                p.join()
    print('num_process: ', num_processes,' pre 用时: ' , time.time() - start_time)
    while len(vocab) < vocab_size: 
        all_tokenized_cnt = 0
        chunk_idx = 0
        idx1, idx2 = max_tuple(total_pair_count, merges, vocab)
        new_idx = original_vocab_len + current_iteration 
        vocab[new_idx] = vocab[idx1] + vocab[idx2]
        # print('第',current_iteration,'轮出现频率最大的', (idx1,idx2), vocab[new_idx], ' 其新索引为', new_idx)
        merges.append((vocab[idx1] , vocab[idx2]))
        new_freq_table: dict[int, dict] = {}
        for chidx, freq_table_split_num in freq_table.items():
            total_pair_count = update_byte_pair_count(freq_table_split_num, total_pair_count, vocab, idx1, idx2, new_idx)
            freq_table_split_num = merge(freq_table_split_num, idx1, idx2, new_idx)
            if check_all_tokenized(freq_table_split_num): all_tokenized_cnt+=1
            new_freq_table[chidx] = freq_table_split_num
        assert total_pair_count[(idx1, idx2)] == 0
        freq_table = new_freq_table
        # if current_iteration % check_period == check_period -1:  
        #     save_checkpoint(checkpoint_path,vocab,merges,total_pair_count,current_iteration,freq_table)
            
        
        current_iteration += 1
        if all_tokenized_cnt == len(freq_table): 
            print('All tokenized!')
            break
        if len(vocab) >= vocab_size: 
            print('Reach maximum vocab size!')
            save_checkpoint(checkpoint_path, vocab, merges, total_pair_count, current_iteration, freq_table)
            return (vocab, merges)
        #if (time.time() - start_time) / 60 % 5 == 1: print(f'已耗时: {(time.time() - start_time) / 60} min')  
    vocabs_without_specials = [word for word in vocab.values() if word != b"<|endoftext|>"]
    for word_bytes in vocabs_without_specials:
        if type(word_bytes) != bytes:
            print('not bytes')
            continue
    return (vocab, merges)

if __name__ == '__main__':
    input_path = '../data/TinyStoriesV2-GPT4-valid.txt'
    #input_path = '../data/TinyStoriesV2-GPT4-train.txt'
    #input_path = '../data/self-test.txt'
    check_period = 500
    # vocab_size = 257 + 2 * check_period
    vocab_size = 500
    special_tokens = ['<|endoftext|>']

    start_time = time.time()
    (vocab, merges) = train_bpe_para(input_path = input_path, vocab_size = vocab_size, special_tokens = special_tokens)
    end_time = time.time()
    print(len(merges), len(set(merges)))
    # print('Final merges: ', merges, end= '\n\n')
    print('程序用时 ', end_time - start_time)

    checkpoint_path = '../data/bpe_checkpoint.pkl'
    checkpoint = load_checkpoint(checkpoint_path)
    vocab = checkpoint['vocab']
    merges = checkpoint['merges']
    total_pair_count = checkpoint["total_pair_count"]
    current_iteration = checkpoint["current_iteration"]
    original_vocab_len = len(vocab)
    freq_table = checkpoint['freq_table']
    # for pair in total_pair_count: 
    #     if total_pair_count[pair] > 0:
    #         print('value != 0: ', pair)
    # for freq_table_split_num in freq_table[0]:
    #     if type(freq_table_split_num) != tuple:
    #         continue
    #     if len(freq_table_split_num) > 0:
    #         print('not fully merged: ', freq_table_split_num)
