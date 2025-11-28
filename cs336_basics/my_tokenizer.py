from collections import defaultdict
import regex as re
import os
from collections.abc import Iterable

# 新增
from collections import defaultdict
import pickle
from typing import Iterable, Iterator
import tempfile
import numpy as np
import tempfile
import multiprocessing
from cs336_basics.pretokenization_example import find_chunk_boundaries

def pre_tokenization_para(chunk: str, chunk_idx: int, COMPILE_PAT, special_tokens:list[str],Queue):
    print('pre_tokenization start, chunk idx: ', chunk_idx)
    pre_tokens = defaultdict(int)
    texts_no_special = re.split("|".join(map(re.escape, special_tokens)), chunk)
    for para in texts_no_special:
        results = re.finditer(COMPILE_PAT, para)
        for i in results:
            word = i.group() # word has form 'str'                    
            word = tuple(word.encode())
            byte_list = list()
            for i in word:
                byte_list.append(bytes([i]))
            word = tuple(byte_list)
            pre_tokens[word] += 1
    Queue.put((chunk_idx, pre_tokens))
    print(f'chunk {chunk_idx} put to queue')

def pre_tokenization_para_pro(
    input_path: str,  # 新增：文件路径（不再传大chunk）
    start: int,       # 新增：当前进程负责的分块起始位置
    end: int,         # 新增：当前进程负责的分块结束位置
    chunk_idx: int,
    COMPILE_PAT,
    special_tokens,
    Queue,
    batch_size: int = 100 * 1024 * 1024  # 小批次大小：100MB（可根据内存调整）
):
    '''仅适用于产生vocab和merges时使用，因为pre_tokens不包含原始语序信息'''
    #print(f"进程 {chunk_idx} 开始处理：{start} ~ {end}（总大小：{(end-start)/1024/1024:.2f} MB）")
    pre_tokens = defaultdict(int)
    remaining = end - start  # 剩余未处理的字节数
    current_pos = start      # 当前读取位置

    # 子进程自行打开文件（避免主进程传递大文件句柄）
    with open(input_path, "rb") as f:
        while remaining > 0:
            # 每次读取“小批次大小”或“剩余字节数”中的较小值（避免超范围）
            read_size = min(batch_size, remaining)
            f.seek(current_pos)  # 移动到当前读取位置
            batch_bytes = f.read(read_size)  # 读取小批次字节（100MB）
            
            if not batch_bytes:
                break  # 极端情况：读取到文件末尾
            
            # 解码并预处理（替换换行符）
            batch_str = batch_bytes.decode("utf-8", errors="ignore")
            batch_str = batch_str.replace("\r\n", "\n").replace("\r", "")
            texts_no_special = re.split("|".join(map(re.escape, special_tokens)), batch_str)
            for para in texts_no_special:
                results = re.finditer(COMPILE_PAT, para)
                for i in results:
                    word = i.group() # word has form 'str'                    
                    word = tuple(word.encode())
                    byte_list = list()
                    for i in word:
                        byte_list.append(bytes([i]))
                    word = tuple(byte_list)
                    pre_tokens[word] += 1
            # 更新进度：已处理字节数 = 原剩余 - 新剩余
            remaining -= read_size
            current_pos += read_size
    Queue.put((chunk_idx, pre_tokens))
    #print(f"进程 {chunk_idx} 已处理：{(end-start-remaining)/1024/1024:.2f} / {(end-start)/1024/1024:.2f} MB")

    #print(f"进程 {chunk_idx} 处理完成")

def pre_tokenization(chunk: str, chunk_idx: int, COMPILE_PAT, special_tokens):
    '''仅适用于产生vocab和merges时使用，因为pre_tokens不包含原始语序信息'''
    pre_tokens = defaultdict(int)
    texts_no_special: list[str] = []
    if special_tokens is not None :
        texts_no_special = re.split("|".join(map(re.escape, special_tokens)), chunk)
    else: texts_no_special.append(chunk)
    for para in texts_no_special:
        # if para == '':
        results = re.finditer(COMPILE_PAT, para)
        for i in results:
            word = i.group() # word has form 'str'                    
            word = tuple(word.encode())
            byte_list = list()
            for i in word:
                byte_list.append(bytes([i]))
            word = tuple(byte_list)
            pre_tokens[word] += 1
    return pre_tokens

class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
        num_processes : int = 4
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = [] if special_tokens == None else special_tokens
        self.num_processes = num_processes
        self.merge_len = len(merges)
        self.vocab_len = len(vocab)
        for special in self.special_tokens:
            special_bytes = special.encode()
            if special_bytes not in vocab.values():
                vocab[self.vocab_len] = special_bytes
                self.vocab_len = self.vocab_len + 1
        
        vocab_inv = defaultdict(int) # 用bytes查找索引
        for idx, my_bytes in vocab.items():
            vocab_inv[my_bytes] = idx
        self.vocab_inv = vocab_inv
        # 预编译正则表达式（移到初始化，避免重复编译）
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.COMPILE_PAT = re.compile(self.PAT, re.UNICODE)
        
        # 预处理特殊token：编码后存入set和dict，加速查找
        self.special_bytes_set = {s.encode() for s in self.special_tokens}
        self.special_bytes_to_token = {s.encode(): [s.encode()] for s in self.special_tokens}
        
        # 预编译merge规则为哈希表（(a, b) → merged_ab）
        self.merge_map = {}
        for idx, merge in enumerate(merges):
            merged_byte = merge[0] + merge[1]
            self.merge_map[merge] = (merged_byte, idx)
    def from_files(self, vocab_filepath, merges_filepath, special_tokens=None):
        vocab = pickle.load(vocab_filepath)
        merges = pickle.load(merges_filepath)
        return Tokenizer(vocab, merges, special_tokens)
    def encode(self, text: str) -> list[int]:
        tokens = []
        pre_tokens = self._pre_tokenize(text)
        
        merged_tokens = []
        for token in pre_tokens:
            # 特殊token直接加入，跳过合并
            if token in self.special_bytes_set:
                merged_tokens.append(self.special_bytes_to_token[token])
                continue
            
            # 普通token进行BPE合并
            token_list = list(token)
            
            restart = True
            while restart:
                restart = False
                candidates = {} # 存放所有相邻字节组合和对应合并结果排序的元组，以merge排序为键，比如{298： ((b'H',b'e'), b'He'), 456: ... }
                for b1, b2 in zip(token_list[:-1], token_list[1:]):
                    tobe_merge = (b1, b2)
                    if tobe_merge not in self.merge_map: continue
                    merged_byte, merge_idx = self.merge_map[tobe_merge]
                    candidates[merge_idx] = (tobe_merge, merged_byte)
                sorted_candidates = sorted(candidates) # sorted_candidates = [41, 161, 1288, 5183]
                if len(sorted_candidates) == 0: break # 如果本轮没有tobe_merge，说明在本merges列表下已经合并完成
                best_merge, after_merge = candidates[sorted_candidates[0]]
                i = 0
                while i < len(token_list) - 1:
                    # 检查当前位置是否有可合并的规则
                    pair = (token_list[i], token_list[i+1])
                    if pair == best_merge:
                        # 执行合并
                        token_list[i] = after_merge
                        # 移除i+1位置的元素（用切片避免内存重排）
                        token_list = token_list[:i+1] + token_list[i+2:]
                        restart = True
                        # 合并后需要从开头重新检查
                        # i = max(0, i - 1)  # 回退一位，避免漏检
                        break
                    else:
                        i += 1
                if restart:
                    continue
            merged_tokens.append(token_list)
        
        for token in merged_tokens:
            tokens.extend(self.vocab_inv[bytes_] for bytes_ in token)
        
        return tokens
    #def _pre_tokenize(self, text: str) -> list[bytes | tuple[bytes, ...]]: 为了IDE不报错，将返回类型删去，不影响运行
    def _pre_tokenize(self, text: str) :
        """优化的预分词函数：减少类型转换，加速特殊token判断"""
        pre_tokens = []
        if not self.special_tokens:
            # 无特殊token时，直接拆分普通文本
            return self._split_ordinary_text(text)
        
        # 拆分特殊token和普通文本（用正则分组保留特殊token）
        sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
        pattern = "|".join(map(re.escape, sorted_specials))
        parts = re.split(f"({pattern})", text)
        
        for part in parts:
            if not part:  # 跳过空字符串
                continue
            # 检查是否为特殊token
            part_bytes = part.encode()
            if part_bytes in self.special_bytes_set:
                pre_tokens.append(part_bytes)
            else:
                # 拆分普通文本为字节元组
                ordinary_tokens = self._split_ordinary_text(part)
                pre_tokens.extend(ordinary_tokens)
        
        return pre_tokens

    def _split_ordinary_text(self, text: str) -> list[tuple[bytes, ...]]:
        """拆分普通文本为字节元组的列表（避免冗余转换）"""
        ordinary_tokens = []
        # 用findall一次性获取所有匹配结果
        matches = self.COMPILE_PAT.findall(text)
        for match in matches:
            # 直接编码为bytes，再拆分为单个字节的元组
            encoded = match.encode()
            byte_tuple = tuple(bytes([b]) for b in encoded)
            ordinary_tokens.append(byte_tuple)
        return ordinary_tokens

    def encode_para_comp(self, input_path: str, start: int, end: int, queue: multiprocessing.Queue, chunk_idx: int = 0, batch_size: int = 8 * 1024 * 1024):
        """子进程函数:分批次处理分块,将token保存为uint16类型的numpy数组(临时文件)"""
        # 创建临时文件（保存当前分块的numpy批次数据，自动删除=False）
        temp_fd, temp_path = tempfile.mkstemp(suffix='.npy', prefix=f'chunk_{chunk_idx}_')
        os.close(temp_fd)  # 关闭文件描述符，后续用numpy读写

        remaining = end - start
        current_pos = start
        print(f"进程 {chunk_idx} 开始处理：{start}~{end}（总大小：{(end-start)/1e6:.2f} MB)")

        with open(input_path, "rb") as f:
            while remaining > 0:
                # 每次读取批次（不超过batch_size或剩余字节）
                read_size = min(batch_size, remaining)
                f.seek(current_pos)
                batch_bytes = f.read(read_size)
                if not batch_bytes:
                    break

                # 预处理：解码+清理换行符
                batch_str = batch_bytes.decode("utf-8", errors="ignore")
                batch_str = batch_str.replace("\r\n", "\n").replace("\r", "")
                
                # 编码当前批次（返回list[int]）
                batch_tokens = self.encode(batch_str)
                if not batch_tokens:  # 无有效token，跳过
                    remaining -= read_size
                    current_pos += read_size
                    continue

                # 转换为uint16类型的numpy数组（节省内存，适配token ID范围）
                # 注意：确保token ID不超过uint16最大值（65535），若超过需改用uint32
                batch_np = np.array(batch_tokens, dtype=np.uint16)

                # 追加写入临时文件（mode='ab'）
                with open(temp_path, 'ab') as tf:
                    np.save(tf, batch_np)

                # 更新进度
                remaining -= read_size
                current_pos += read_size
                processed_mb = (end - start - remaining) / 1e6
                total_mb = (end - start) / 1e6
                print(f"进程 {chunk_idx} 进度：{processed_mb:.2f}/{total_mb:.2f} MB")

        # 子进程仅返回分块索引和临时文件路径（不传递大数组）
        queue.put((chunk_idx, temp_path))
        print(f"进程 {chunk_idx} 处理完成，临时文件：{temp_path}")

    def encode_para(self, file_path: str, num_chunk: int, output_path: str) -> None:
        """
        并行处理超大文件，最终保存为uint16类型的numpy数组（.npy格式）
        
        参数：
            file_path: 输入txt文件路径
            num_chunk: 分块数量
            output_path: 最终numpy数组保存路径（.npy后缀）
        """
        # 计算分块边界（基于分隔符b'<|endoftext|>'）
        # 限制并发进程数为CPU核心数（避免资源竞争）
        max_proc = max(num_chunk, os.cpu_count() or 4)
        print('max_proc: ', max_proc)
        with open(file_path, 'rb') as f:
            boundaries = find_chunk_boundaries(f, max_proc, b'<|endoftext|>')
        print(f"文件分块完成，共 {len(boundaries)-1} 个分块")

        # 启动子进程处理每个分块
        proc_list = []
        queue = multiprocessing.Queue()
        chunk_idx = 0
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            p = multiprocessing.Process(
                target=self.encode_para_comp, 
                args=(file_path, start, end, queue, chunk_idx)
            )
            p.start()
            proc_list.append(p)
            chunk_idx += 1
        temp_paths = {}
        c = 0
        while c < len(proc_list):
            c_idx, temp_path = queue.get()
            temp_paths[c_idx] = temp_path
            print(f"已接收分块 {c_idx} 的临时文件, 进度 {c+1}/{len(proc_list)} 块")
            c += 1

        for p in proc_list:
            p.join()
            p.close()  # 释放进程资源

        # 流式合并所有临时文件（不加载全量数据到内存）
        print(f"开始合并 {len(temp_paths)} 个分块的临时文件...")
        with open(output_path, 'wb') as out_f:
            # 按分块索引顺序合并（确保token顺序正确）
            for i in sorted(temp_paths.keys()):
                temp_path = temp_paths[i]
                #print(f"合并分块 {i}：{temp_path}")
                
                # 读取临时文件中的所有numpy批次并写入最终文件
                with open(temp_path, 'rb') as tf:
                    while True:
                        try:
                            # 逐个读取批次numpy数组
                            batch_np = np.load(tf, allow_pickle=False)
                            np.save(out_f, batch_np)
                        except EOFError:
                            break  
                
                # 删除临时文件（释放磁盘空间）
                os.remove(temp_path)
                #print(f"分块 {i} 合并完成，已删除临时文件")
        print(f"所有分块处理完成，最终结果保存至：{output_path}")

        print("开始合并为单个数组...")
        all_batches = []
        with open(output_path, 'rb') as f:
            while True:
                try:
                    all_batches.append(np.load(f, allow_pickle=False))
                except EOFError:
                    break
        final_np = np.concatenate(all_batches, axis=0)
        np.save(output_path.replace('.npy', '_merged.npy'), final_np)
        print(f"单个数组合并完成，保存至：{output_path.replace('.npy', '_merged.npy')}")

        # print(f"所有分块处理完成，最终结果保存至：{output_path}")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Lazily encodes an iterable of strings into token IDs, yielding tokens one by one.
        
        Args:
            iterable: An iterable of strings (e.g., file handle, generator) to encode.
            
        Yields:
            Integer token IDs generated from the input iterable.
        """
        for text in iterable:
            # 对每个字符串调用已实现的encode方法，逐个yield token ID
            for token_id in self.encode(text):
                yield token_id


    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs back into a human-readable string.
        
        Args:
            ids: A list of integer token IDs.
            
        Returns:
            The decoded string representation of the input token IDs.
        """
        full_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        return full_bytes.decode("utf-8", errors="replace")

if __name__ == "__main__":
    import os
    import re
    import numpy as np
    owt_train_token_path = './data/owt_train_token.npy'
    # 目标文件夹路径
    folder_path = r'C:\Users\Leo\Desktop\123'

    # 正则表达式：匹配"chunk_数字_"的结构（提取编号）
    pattern = re.compile(r'chunk_(\d+)_')

    # 初始化字典
    temp_paths = {}

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.npy'):  # 只处理npy文件
            # 提取chunk编号（如"chunk_2_..." → 编号2）
            match = pattern.search(filename)
            if match:
                chunk_num = int(match.group(1))  # 提取并转为整数
                # 拼接完整文件路径
                file_path = os.path.join(folder_path, filename)
                # 存入字典
                temp_paths[chunk_num] = file_path

    # 打印结果
    for chunk_num, path in temp_paths.items():
        print(f"chunk_{chunk_num}: {path}")
    print(f"开始合并 {len(temp_paths)} 个分块的临时文件...")
    with open(owt_train_token_path, 'wb') as out_f:
        # 按分块索引顺序合并（确保token顺序正确）
        for i in sorted(temp_paths.keys()):
            temp_path = temp_paths[i]
            #print(f"合并分块 {i}：{temp_path}")
            
            # 读取临时文件中的所有numpy批次并写入最终文件
            with open(temp_path, 'rb') as tf:
                while True:
                    try:
                        # 逐个读取批次numpy数组
                        batch_np = np.load(tf, allow_pickle=False)
                        # 追加写入最终文件
                        np.save(out_f, batch_np)
                    except EOFError:
                        break  # 临时文件读取完毕
            
            # 删除临时文件（释放磁盘空间）
            # os.remove(temp_path)
            #print(f"分块 {i} 合并完成，已删除临时文件")
    print(f"所有分块处理完成，最终结果保存至：{owt_train_token_path}")

    print("开始合并为单个数组（可能耗时较长）...")
    all_batches = []
    with open(owt_train_token_path, 'rb') as f:
        while True:
            try:
                all_batches.append(np.load(f, allow_pickle=False))
            except EOFError:
                break
    final_np = np.concatenate(all_batches, axis=0)
    np.save(owt_train_token_path.replace('.npy', '_merged.npy'), final_np)
    print(f"单个数组合并完成，保存至：{owt_train_token_path.replace('.npy', '_merged.npy')}")

    # 读取合并前的文件
    batches = []
    with open(owt_train_token_path, 'rb') as f:
        while True:
            try:
                batch = np.load(f, allow_pickle=False)
                batches.append(batch)
            except EOFError:
                break

    # # 查看每个批次的结构
    # for i, batch in enumerate(batches):
    #     print(f"合并前 - 批次{i+1}: shape={batch.shape}, data={batch}, dtype={batch.dtype}")
    
    t_token = np.load(owt_train_token_path.replace('.npy', '_merged.npy'))
    print(len(t_token))
    