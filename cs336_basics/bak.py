s = b' a '
bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
print('length of s: ', len(s))
new_s = ''
for byte in s:
    # byte 自动变成 对应单字节的ord值
    # print('original: ', chr(byte).encode())
    if byte not in bs: 
        byte = byte + 2**8
    # print('after: ', chr(byte).encode())
    new_s += chr(byte) 

print('s after: ',new_s)
print('s after: ',new_s.encode())
