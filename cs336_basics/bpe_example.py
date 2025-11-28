
def pre_tokens2leteters(pre_tokens: dict, vocab: dict):
    letters = {}
    for tup in pre_tokens:
        t = tuple()
        for item in tup:
            if type(item) == int: t += ((vocab[item]).decode(),)
            if type(item) == bytes: t += (item.decode(),)
        letters[t] = pre_tokens[tup]
    return letters

text = '''low low low low low
lower lower widest widest widest
newest newest newest newest newest newest'''

text = '''the cat ate'''

text = '''One day, a little girl named Lily found a needle in her room. She knew it was difficult to play with it because it was sharp. Lily wanted to share the needle with her mom, so she could sew a button on her shirt. Lily went to her mom and said, "Mom, I found this needle. Can you share it with me and sew my shirt?" Her mom smiled and said, "Yes, Lily, we can share the needle and fix your shirt." Together, they shared the needle and sewed the button on Lily's shirt. It was not difficult for them because they were sharing and helping each other. After they finished, Lily thanked her mom for sharing the needle and fixing her shirt. They both felt happy because they had shared and worked together.'''
# step 1: vocab initialization
# indices = list(map(int, text.encode("utf-8")))  # @inspect indices
merges: dict[tuple[int, int], int] = {}  # index1, index2 => merged index
vocab: dict = {x+1: bytes([x]) for x in range(256)}  # index -> bytes
vocab[0] = '<|endoftext|>' 
offset = 1 # 这一项是因为我们人为在字节形成的字典前面加了一个EOT的token
original_vocab_len = len(vocab)

# step 2: pre-tokenization
pre_tokens = {} # {(108,111): 5, (123,112):4}
words = {} 
split_by_space = text.split()
split_by_space[1:] = [' '+ i for i in split_by_space[1:]] # 将单词前的空格划拨给这个单词，在bpe_example中不必做这一步
for item in split_by_space:
    if item in words:
        words[item] += 1
    else:
        words[item] = 1
print(words)
for item in words:
    letter_tuple = tuple(item) # ('l','o','w')
    index_tuple = tuple() # (108, 111, 119)
    for letter in letter_tuple:
        index_tuple += (ord(letter)+offset,) # 添加单个元素进元组需要加逗号。为了正确映射字节到字符，需要加上offset
    pre_tokens[index_tuple] = words[item] #{(108, 111, 119): 5, (108, 111, 119, 101, 114): 2, (119, 105, 100, 101, 115, 116): 3, (110, 101, 119, 101, 115, 116): 6}
print(pre_tokens)

# step 3: merge
i = 0 
max_num_merge = 20
all_tokenized = False
while i < max_num_merge and not all_tokenized:
    byte_pair_count = {}
    for word_split_tuple in pre_tokens: # 只在pre_tokens内部的某一个元组，而非跨元组进行BPE
        for index1, index2 in zip(word_split_tuple[0:], word_split_tuple[1:]):
            if (index1, index2) in byte_pair_count: # 防止每次都遍历所有字符。这样，对于'lo',只需要在'low'中查找一次就可以使'lo'的count+5
                byte_pair_count[(index1, index2)] += pre_tokens[word_split_tuple]
            else:
                byte_pair_count[(index1, index2)] = pre_tokens[word_split_tuple]
    print(byte_pair_count)
    print(pre_tokens2leteters(byte_pair_count,vocab))
    # 查找本轮出现最多的字节对
    max_count = max(byte_pair_count.values()) # 先按count的值，选出最多的元组序列
    candidates = [pair for pair in byte_pair_count if byte_pair_count[pair] == max_count] 
    # 元组比较规则：先比第一个元素，若相同则比第二个元素
    max_tuple = max(candidates) # 再在这些元组中选择元组内数字最大的那个

    idx1, idx2 = max_tuple # 将本轮count值最大的元组中的两个token提取出来
    # 为该字节对分配新的token并添加进vocab（这就一定不在byte 0-255的范围了）
    new_idx = original_vocab_len + i 
    vocab[new_idx] = vocab[idx1] + vocab[idx2]
    print('本轮出现频率最大的字节对及其对应的字节序列及其解码结果', max_tuple, vocab[new_idx], vocab[new_idx].decode())
    i += 1
    # 更新merges(dict[tuple[int, int], int])字典（调试用）
    merges[(vocab[idx1] , vocab[idx2])] = new_idx
    # 替换掉所有word_split_tuple中的(idx1, idx2)模式
    new_pre_tokens = {}
    for word_split_tuple in pre_tokens:
        j = 0
        new_word_split_tuple = tuple() 
        while j < len(word_split_tuple): # 仿照课件写法
            if j < len(word_split_tuple) - 1 and word_split_tuple[j] == idx1 and word_split_tuple[j+1] == idx2: # word_split_tuple 里元组存的都是token值，而非字符本身，所以应该与idx1,2比而不是跟vocab[idx1,2]比
                new_word_split_tuple += (new_idx,)
                j += 2
            else:
                new_word_split_tuple += (word_split_tuple[j],)
                j += 1
        new_pre_tokens[new_word_split_tuple] = pre_tokens[word_split_tuple]
    pre_tokens = new_pre_tokens
    # 当所有元组内只剩下一个元素时，说明所有内容已经被纳入vocab，这时候就应该停止BPE
    # 当然也可以认为设置max_num_merge，这样vocab就不会运行到纳入所有完整单词
    cnt = 0
    for tup in pre_tokens:
        if len(tup) == 1: cnt += 1 
    if cnt == len(pre_tokens): all_tokenized = True
    print('经过merge后的pre_token:', pre_tokens)
    print('对应的字符为', pre_tokens2leteters(pre_tokens, vocab))



print('=========================== final results ===========================')
added = {k: v for k, v in filter(lambda item: item[0] > original_vocab_len - 1, vocab.items())}
print('final vocab (only the added part): ', added, '\n')
print('num of merges used: ', i, '\n')
print('final merges dict: ', pre_tokens2leteters(merges,vocab), '\n')
print('final tokens: ', pre_tokens, '\n')
print('final letters decoded from tokens: ', pre_tokens2leteters(pre_tokens,vocab), '\n')

