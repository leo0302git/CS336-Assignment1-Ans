# def has_subtuple(main_tuple, sub_tuple):
#     cnt = 0
#     sub_len = len(sub_tuple)
#     main_len = len(main_tuple)
#     # 若子元组长度大于主元组，直接返回 False
#     if sub_len > main_len:
#         return False
#     # 遍历主元组中可能的起始位置
#     for i in range(main_len - sub_len + 1):
#         # 截取主元组中长度为 sub_len 的连续元素，与子元组比较
#         if main_tuple[i:i+sub_len] == sub_tuple:
#             cnt += 1
#     return cnt
# def init_byte_pair_count(freq_table_split_num):
#     byte_pair_count: dict[tuple, int] = {}
#     for word_split_tuple in freq_table_split_num: # 只在pre_tokens内部的某一个元组，而非跨元组进行BPE
#         for index1, index2 in zip(word_split_tuple[0:], word_split_tuple[1:]):
#             byte_pair_count[(index1, index2)] = byte_pair_count.get((index1, index2), 0) + freq_table_split_num[word_split_tuple]
#     return byte_pair_count
# def count_pair(tup, tupnum):
#     pair = {}
#     for index1, index2 in zip(tup[0:], tup[1:]):
#         pair[(index1, index2)] = pair.get((index1, index2), 0) + tupnum
#     return pair

# def merge_tup(tup, idx1, idx2, new_idx):
#     j = 0
#     new_tup = tuple()
#     while j < len(tup): # 仿照课件写法
#             if j < len(tup) - 1 and tup[j] == idx1 and tup[j+1] == idx2:
#                 new_tup += (new_idx,)
#                 j += 2
#             else:
#                 new_tup += (tup[j],)
#                 j += 1
#     return new_tup

# def pair_changed(tup, max_tup, tupnum, new_idx):
#     idx1, idx2 = max_tup
#     broken_pair = count_pair(tup, tupnum)
#     new_tup = merge_tup(tup, idx1, idx2, new_idx)
#     add_pair = count_pair(new_tup, tupnum)
#     return broken_pair, add_pair

# idx1, idx2 = (13,10)
# freq_table_split_num = {(13, 10): 1, (13,10,9,13,10,13,10,13): 3}
# byte_pair_count = {(13,10): 10, (10,13): 6}

# idx1, idx2 = (309,309)
# freq_table_split_num = {(309,309,309,13):1}
# byte_pair_count = {(309,309): 2, (309,13): 1}

# max_tuple = (idx1, idx2)
# broken_pair={}
# add_pair = {}
# new_idx = -1
# self_add_pair_cnt = 0

# for tup in freq_table_split_num:
#     if has_subtuple(tup, (idx1, idx2)) > 0:
#         broken_pair, add_pair = pair_changed(tup, (idx1, idx2), freq_table_split_num[tup], new_idx)

# print('max_tuple',max_tuple,end='\n')
# print('freq_table_split_num', freq_table_split_num, end='\n')
# print('byte_pair_count', byte_pair_count, end='\n')

# print('broken_pair', broken_pair, end='\n')
# print('add_pair', add_pair, end='\n')


# # print('broken mask', broken_mask, end='\n')
# # print('add mask', add_mask, end='\n')

# # max_tuple (13, 10)
# # freq_table_split_num {(13, 10): 1, (13, 10, 9, 13, 10, 13, 10, 13): 3}
# # byte_pair_count {(13, 10): 10, (10, 13): 6}
# # broken_pair {(13, 10): 10, (10, 9): 3, (9, 13): 3, (10, 13): 6}
# # add_pair {(-1, 9): 3, (-1, -1): 3, (9, -1): 3, (-1, 13): 3}

