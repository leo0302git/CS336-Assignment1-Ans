import pickle
from tests.adapters import Tokenizer
import time
def extract_documents(
    file_path: str,
    split_special_token: str,
    num_docs: int,
    max_read_size: int = 10 * 1024 * 1024,  # 单次最大读取10MB
    encoding: str = "utf-8"
) -> list[str]:
    """
    从大文件中提取指定数量的文档（分块读取，单次最大10MB）
    
    参数：
        file_path: txt文件路径
        split_special_token: 文档分隔符（字符串）
        num_docs: 需要提取的文档数量
        max_read_size: 单次最大读取字节数（默认10MB）
        encoding: 文件编码
    """
    docs = []  # 存储提取的文档
    remaining = ""  # 记录上一块未完成的内容（跨块的文档）
    sep = split_special_token
    sep_len = len(sep)  # 分隔符长度，用于处理边界
    
    with open(file_path, "r", encoding=encoding) as f:
        while len(docs) < num_docs:
            # 1. 分块读取（最多读取 max_read_size 字节）
            # 注意：read(n) 按字符数读取，这里通过限制字节数间接控制
            chunk = f.read(max_read_size)
            if not chunk:  # 文件已读完
                break
            
            # 2. 拼接上一次剩余的内容，形成完整的处理块
            content = remaining + chunk
            remaining = ""  # 重置剩余内容
            
            # 3. 按分隔符分割，提取完整文档
            # 避免直接 split(sep) 导致的性能问题（大字符串split耗时）
            start = 0
            while start <= len(content) - sep_len:
                # 查找分隔符位置
                pos = content.find(sep, start)
                if pos == -1:  # 块内未找到完整分隔符，剩余内容留到下次处理
                    remaining = content[start:]
                    break
                
                # 提取分隔符前的文档（从start到pos）
                doc = content[start:pos].strip()
                if doc:  # 过滤空文档
                    docs.append(doc)
                    if len(docs) >= num_docs:  # 达到目标数量，提前退出
                        remaining = ""
                        return docs[:num_docs]
                
                # 移动到下一个分隔符之后
                start = pos + sep_len
        
        # 4. 处理文件结束后剩余的最后一个文档
        if remaining.strip() and len(docs) < num_docs:
            docs.append(remaining.strip())
    
    return docs[:num_docs]

if __name__ == '__main__':
    special_tokens = ["<|endoftext|>"]
    OpenWebText_vocab_path = './data/OpenWebText_vocab_32000.pkl'
    OpenWebText_merges_path = './data/OpenWebText_merges_32000.pkl'
    TinyStories_vocab_path = './data/tinystories_vocab_10000.pkl'
    TinyStories_merges_path = './data/tinystories_merges_10000.pkl'
    with open(OpenWebText_vocab_path, "rb") as f:
        owt_vocab = pickle.load(f)
    with open(OpenWebText_merges_path, "rb") as f:
        owt_merges = pickle.load(f)
    with open(TinyStories_vocab_path, "rb") as f:
        tiny_vocab = pickle.load(f)
    with open(TinyStories_merges_path, "rb") as f:
        tiny_merges = pickle.load(f)
    
    owt_tokenizer = Tokenizer(vocab=owt_vocab, merges=owt_merges,special_tokens=special_tokens,num_processes=4)
    tiny_tokenizer = Tokenizer(vocab=tiny_vocab, merges=tiny_merges,special_tokens=special_tokens,num_processes=4)

    tinysoties_path = './data/TinyStoriesV2-GPT4-valid.txt'
    owt_path = './data/owt_valid.txt'
    tiny_docs = extract_documents(tinysoties_path,special_tokens[0],10)
    owt_docs = extract_documents(owt_path,special_tokens[0],10)

    tokenizers = [tiny_tokenizer, owt_tokenizer]
    tokenizers_name = ['tinystories_tokenizer', 'owt_tokenizer']
    docs = [tiny_docs, owt_docs]  # docs: list[list[str]]（假设长度与 tokenizers 一致）
    tokens: list[list] = [] 
    
    # task 1
    print('Task 1')
    # 仅在第一遍没有生成token的时候运行，后面可以直接load token
    for tokenizer_idx, t in enumerate(tokenizers):
        doc_tokens = []
        # 遍历当前 tokenizer 对应的文档列表（docs[tokenizer_idx]）
        for story in docs[tokenizer_idx]:
            doc_tokens.append(t.encode(story))
            # 将当前文档列表的编码结果添加到中层列表
        tokens.append(doc_tokens)
    dump_path = './data/tokenizer_experiment_task1.pkl'
    with open(dump_path, 'wb') as f:
        pickle.dump(tokens,f)
    load_path = './data/tokenizer_experiment_task1.pkl'
    with open(load_path, 'rb') as f:
        tokenizer_experiment_task1_token = pickle.load(f)
    compression_rate = [-1.0, -1.0]
    for i, tokens in enumerate(tokenizer_experiment_task1_token):
        total_token_len = 0
        total_bytes = 0
        for j, token in enumerate(tokens):
            total_token_len += len(token)
            total_bytes += len(docs[i][j])
        compression_rate[i] = total_bytes / total_token_len
        print('tokenizers_name: ', tokenizers_name[i], '\ncompression_rate: ', compression_rate[i])
    
    # Task 2
    print('Task 2')
    cross_tokens = []
    for owt_doc in owt_docs:
        cross_tokens.append(tiny_tokenizer.encode(owt_doc))
    total_token_len = 0
    total_bytes = 0
    for i, token in enumerate(cross_tokens):
        total_token_len += len(token)
        total_bytes += len(owt_docs[i])
    compression_rate_task2 = total_bytes / total_token_len
    print('compression_rate: ', compression_rate_task2)
    pass


    print('Task 4')
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

    # 路径配置
    ts_train_path = './data/TinyStoriesV2-GPT4-train.txt'
    ts_valid_path = './data/TinyStoriesV2-GPT4-valid.txt'
    owt_train_path = './data/owt_train.txt'
    owt_valid_path = './data/owt_valid.txt'

    # 输出路径（.npy格式，保存uint16类型numpy数组）
    ts_valid_token_path = './data/tinystories_valid_token.npy'
    ts_train_token_path = './data/tinystories_train_token.npy'
    owt_valid_token_path = './data/owt_valid_token.npy'
    owt_train_token_path = './data/owt_train_token.npy'

    num_chunks = 10  # 分块数量（可根据CPU核心数调整）

    # TinyStories valid 处理
    start_time = time.time()
    tiny_tokenizer.encode_para(ts_valid_path, num_chunks, ts_valid_token_path)
    ts_valid_done = time.time()
    print(f'ts_valid处理耗时:{ts_valid_done - start_time:.2f} s')

    # TinyStories train 处理
    tiny_tokenizer.encode_para(ts_train_path, num_chunks, ts_train_token_path)
    ts_train_done = time.time()
    print(f'ts_train处理耗时:{ts_train_done - ts_valid_done:.2f} s')

    # OWT valid 处理
    owt_tokenizer.encode_para(owt_valid_path, num_chunks, owt_valid_token_path)
    owt_valid_done = time.time()
    print(f'owt_valid处理耗时:{owt_valid_done - ts_train_done:.2f} s')

    # OWT train 处理（22GB大文件）
    owt_tokenizer.encode_para(owt_train_path, num_chunks, owt_train_token_path)
    owt_train_done = time.time()
    print(f'owt_train处理耗时:{owt_train_done - owt_valid_done:.2f} s')

    print(f'ts_valid处理耗时:{ts_valid_done - start_time:.2f} s')
    print(f'ts_train处理耗时:{ts_train_done - ts_valid_done:.2f} s')
    print(f'owt_valid处理耗时:{owt_valid_done - ts_train_done:.2f} s')
    print(f'owt_train处理耗时:{owt_train_done - owt_valid_done:.2f} s')

    




    # test encode_para on se;f-test.txt
    # ts_train_token_done = time.time()
    # self_test_token = tiny_tokenizer.encode_para(self_test_path, 10)
    # decode_self_txt = tiny_tokenizer.decode(self_test_token)
    # with open(self_test_path, 'r', encoding= 'utf-8') as f:
    #     self_txt = f.read()
    # assert decode_self_txt == self_txt
    # with open(ts_valid_token_path, 'wb') as f:
    #     pickle.dump(self_test_token, f)
    # ts_valid_token_done = time.time()
    # print('self_test_token_done (s): ', ts_valid_token_done - ts_train_token_done)