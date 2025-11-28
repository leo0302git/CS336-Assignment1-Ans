from train_bpe import load_checkpoint, check_all_tokenized

if __name__ == '__main__':
    # for i in range(0,256):
    #     print(i, chr(i))
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    # now get the representations of the other 68 integers that do need shifting
    # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
    # Get printable representations of the remaining integers 68 integers.
    n = 0
    for b in range(2**8):
        if b not in bs:
            # If this integer isn't in our list of visually-representable
            # charcters, then map it to the next nice character (offset by 256)
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))

    checkpoint_path = './data/bpe_checkpoint.pkl'
    checkpoint = load_checkpoint(checkpoint_path)
    vocab = checkpoint['vocab']
    merges = checkpoint['merges']
    total_pair_count = checkpoint["total_pair_count"]
    current_iteration = checkpoint["current_iteration"]
    original_vocab_len = len(vocab)
    freq_table = checkpoint['freq_table']
    vocabs_without_specials = [word for word in vocab.values() if word != b"<|endoftext|>"]
    for word_bytes in vocabs_without_specials:
        if type(word_bytes) is not bytes:
            print(word_bytes)
        assert b"<|" not in word_bytes
    # for pair in total_pair_count: 
    #     if total_pair_count[pair] > 0:
    #         print('value != 0: ', pair)
    # for freq_table_split_num in freq_table[0]:
    #     if type(freq_table_split_num) != tuple:
    #         # print('not dict!')
    #         continue
    #     if len(freq_table_split_num) > 0:
    #         print('not fully merged: ', freq_table_split_num)
    pass