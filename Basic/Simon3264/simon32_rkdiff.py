import numpy as np
from os import urandom
import gc

def WORD_SIZE():
    return(16)

MASK_VAL = 2 ** WORD_SIZE() - 1

const = [0xfffd, 0xfffd, 0xfffd, 0xfffd,
         0xfffd, 0xfffc, 0xfffd, 0xfffc,
         0xfffc, 0xfffc, 0xfffd, 0xfffc,
         0xfffc, 0xfffd, 0xfffc, 0xfffd,
         0xfffc, 0xfffd, 0xfffd, 0xfffc,
         0xfffc, 0xfffc, 0xfffc, 0xfffd,
         0xfffd, 0xfffd, 0xfffc, 0xfffc]


def rol(x,k):
    return(((x << k) & MASK_VAL) | (x >> (WORD_SIZE() - k))) 


def ror(x,k):
    return((x >> k) | ((x << (WORD_SIZE() - k)) & MASK_VAL)) 


def enc_one_round_simon(p, k):
    c1 = p[0] 
    c0 = (rol(p[0], 8) & rol(p[0],1)) ^ rol(p[0],2) ^ p[1] ^ k
    return(c0,c1)

def dec_one_round_simon(c, k):
    c0, c1 = c[0], c[1]
    c0 = ((c0 ^ k[0]) ^ rol(c1, 2)) ^ (rol(c1, 1) & rol(c1, 8))
    return c1, c0


def expand_key_simon_reversed(k, t):
    if t < 4:
        res = []
        for i in range(t):
            res.append(k[i])
        return list(reversed(res))
    
    ks = [0 for i in range(t)]
    ks[0] = k[0]
    ks[1] = k[1]
    ks[2] = k[2]
    ks[3] = k[3]
    # print(ks)
    for i in range(t - 4):
        tmp = ror(ks[i+1],3) ^ ks[i+3]
        # print(i,t-5-i)
        ks[i+4] = const[32-5-i] ^ ks[i] ^ tmp ^ ror(tmp,1)
    return list(reversed(ks))


def expand_key_simon(k, t):
    if t < 4:
        res = []
        for i in range(t):
            res.append(k[3-i])
        return res
    ks = [0 for i in range(t)]
    ks[0] = k[3]
    ks[1] = k[2]
    ks[2] = k[1]
    ks[3] = k[0]
    for i in range(t - 4):
        tmp = ror(ks[i+3],3) ^ ks[i+1]
        ks[i+4] = const[i] ^ ks[i] ^ tmp ^ ror(tmp,1)
    return(ks)



def expand_key_simon_all(k, t):
    keys = [k]
    if t < 4:
        res = []
        for i in range(t):
            res.append(k[3-i])
        return res
    ks = [0 for i in range(t)]
    ks[0] = k[3]
    ks[1] = k[2]
    ks[2] = k[1]
    ks[3] = k[0]
    for i in range(t - 4):
        tmp = ror(ks[i+3],3) ^ ks[i+1]
        ks[i+4] = const[i] ^ ks[i] ^ tmp ^ ror(tmp,1)
    return(ks)

# def decrypt_simon(p, ks):
#     x, y = p[0], p[1]
#     for k in ks:
#         x,y = enc_one_round_simon((x,y), k)
#     return(x, y)


def encrypt_simon(p, ks):
    x, y = p[0], p[1]
    for k in ks:
        x,y = enc_one_round_simon((x,y), k)
    return(x, y)


def decrypt_simon(c, ks):
    x, y = c[0], c[1]
    for k in reversed(ks):
        x,y = dec_one_round_simon((x,y), k)
    return(x, y)

def convert_to_binary(arr,s_groups=1):
  X = np.zeros((8 * WORD_SIZE() * s_groups,len(arr[0])),dtype=np.uint8) 
  for i in range(8 * WORD_SIZE() * s_groups):
    index = i // WORD_SIZE() 
    offset = WORD_SIZE() - (i % WORD_SIZE()) - 1
    X[i] = (arr[index] >> offset) & 1
  X = X.transpose() 
  return(X)


def make_train_data(n, nr, diff=(0x0,0x40),key_diff=(0x0,0x0,0x0,0x0),s_groups=1):
    Y = np.frombuffer(urandom(n), dtype=np.uint8)
    Y = Y & 1
    keys = np.frombuffer(urandom(8 * n), dtype=np.uint16).reshape(4, -1)
    keys_diff = np.array([keys[0] ^ key_diff[0], keys[1] ^ key_diff[1], keys[2] ^ key_diff[2], keys[3] ^ key_diff[3]],dtype=np.uint16)
    num_rand_samples = np.sum(Y == 0)
    ks = expand_key_simon(keys, nr)
    ks_diff = expand_key_simon(keys_diff, nr)
    del keys,keys_diff;gc.collect()

    X_result = []
    
    
    
    for i in range(s_groups):
        plain0l = np.frombuffer(urandom(2 * n), dtype=np.uint16)
        plain0r = np.frombuffer(urandom(2 * n), dtype=np.uint16)
        plain1l = plain0l ^ diff[0]
        plain1r = plain0r ^ diff[1]
        plain1l[Y == 0] = np.frombuffer(urandom(2 * num_rand_samples), dtype=np.uint16)
        plain1r[Y == 0] = np.frombuffer(urandom(2 * num_rand_samples), dtype=np.uint16)
        ctdata0l, ctdata0r = encrypt_simon((plain0l, plain0r), ks)
        ctdata1l, ctdata1r = encrypt_simon((plain1l, plain1r), ks_diff)
        
        delta_ctdata0l = ctdata0l ^ ctdata1l
        delta_ctdata0r = ctdata0r ^ ctdata1r
        
        # delta_ctdata0rr = ctdata0l ^ ctdata1l ^ ctdata0r ^ ctdata1r
        # delta_ctdata0 = ctdata0l ^ ctdata0r
        # delta_ctdata1 = ctdata1l ^ ctdata1r
        # secondLast_ctdata0r = rol(ctdata0r, 8) & rol(ctdata0r, 1) ^ rol(ctdata0r, 2) ^ ctdata0l
        # secondLast_ctdata1r = rol(ctdata1r, 8) & rol(ctdata1r, 1) ^ rol(ctdata1r, 2) ^ ctdata1l
        
        
 
        secondLast_ctdata0r = rol(ctdata0r, 8) & rol(ctdata0r, 1) ^ rol(ctdata0r, 2) ^ ctdata0l
        secondLast_ctdata1r = rol(ctdata1r, 8) & rol(ctdata1r, 1) ^ rol(ctdata1r, 2) ^ ctdata1l
 
        delta_secondLast_ctdata0r =  secondLast_ctdata0r ^ secondLast_ctdata1r
        
        
        thirdLast_ctdata0r = ctdata0r ^ rol(secondLast_ctdata0r,8) & rol(secondLast_ctdata0r,1) ^ rol(secondLast_ctdata0r,2)
        thirdLast_ctdata1r = ctdata1r ^ rol(secondLast_ctdata1r,8) & rol(secondLast_ctdata1r,1) ^ rol(secondLast_ctdata1r,2)
        
        
        delta_thirdLast_ctdata0r = thirdLast_ctdata0r ^ thirdLast_ctdata1r


        X_result.append(delta_ctdata0l)
        X_result.append(delta_ctdata0r)

        X_result.append(ctdata0l)
        X_result.append(ctdata0r)
        
        X_result.append(ctdata1l)
        X_result.append(ctdata1r)

        #X_result.append(secondLast_ctdata0r)
        #X_result.append(secondLast_ctdata1r)
        
        X_result.append(delta_secondLast_ctdata0r)
        X_result.append(delta_thirdLast_ctdata0r)
        del plain0l,plain0r,plain1l,plain1r,ctdata0l,ctdata0r,ctdata1l,ctdata1r,delta_ctdata0l,delta_ctdata0r,secondLast_ctdata0r,secondLast_ctdata1r,delta_thirdLast_ctdata0r
        gc.collect()
    del ks,ks_diff;gc.collect()    
    X= convert_to_binary(X_result,s_groups=s_groups)
    #X = np.tile(X,s_groups)
    return (X, Y)


#X, Y = make_train_data(10,10,s_groups=1)

def getdiffs(nr, n=10**4, diff=(0x0,0x40),key_diff=(0x0,0x0,0x0,0x0)):
    
    keys = np.frombuffer(urandom(8 * n), dtype=np.uint16).reshape(4, -1)
    keys_diff = np.array([keys[0] ^ key_diff[0], keys[1] ^ key_diff[1], keys[2] ^ key_diff[2], keys[3] ^ key_diff[3]],dtype=np.uint16)
    ks = expand_key_simon(keys, nr)
    ks_diff = expand_key_simon(keys_diff, nr)
    del keys,keys_diff;gc.collect()
    key_outdiffs = np.array(ks) ^ np.array(ks_diff)
    # print(outdiffs)
    key_outdiffs = np.transpose(key_outdiffs)
    key_outdiffs_res = []
    for key_outdiff in key_outdiffs:
        key_outdiffs_res.append("_".join([f"{hex(item)}" for item in key_outdiff])) 
        
    plain0l = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    plain0r = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    plain1l = plain0l ^ diff[0]
    plain1r = plain0r ^ diff[1]
    ctdata0l, ctdata0r = encrypt_simon((plain0l, plain0r), ks)
    ctdata1l, ctdata1r = encrypt_simon((plain1l, plain1r), ks_diff)
    
    delta_ctdata0l = ctdata0l ^ ctdata1l
    delta_ctdata0r = ctdata0r ^ ctdata1r
    outdiffs = delta_ctdata0l*(2**16)+delta_ctdata0r
    
    del plain0l,plain0r,plain1l,plain1r,ctdata0l,ctdata0r,ctdata1l,ctdata1r,delta_ctdata0l,delta_ctdata0r
    gc.collect()
    del ks,ks_diff;gc.collect()    
    return outdiffs,key_outdiffs_res


def getdiffs_reversed(nr, n=10**4, diff=(0x0,0x40),key_diff=(0x0,0x0,0x0,0x0)):
    print("Round:",nr)
    keys = np.frombuffer(urandom(8 * n), dtype=np.uint16).reshape(4, -1)
    keys_diff = np.array([keys[0] ^ key_diff[0], keys[1] ^ key_diff[1], keys[2] ^ key_diff[2], keys[3] ^ key_diff[3]],dtype=np.uint16)
    # print(keys)
    # print(keys_diff)
    # print(keys_diff.shape)
    ks = expand_key_simon_reversed(keys, nr)
    ks_diff = expand_key_simon_reversed(keys_diff, nr)
    del keys,keys_diff;gc.collect()
    # print(ks)
    # print(ks_diff)
    key_indiffs = np.array(ks) ^ np.array(ks_diff)
    # print(indiffs)
    key_indiffs = np.transpose(key_indiffs)
    key_indiffs_res = []
    for key_outdiff in key_indiffs:
        key_indiffs_res.append("_".join([f"{hex(item)}" for item in key_outdiff])) 
        
    ctdata0l = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    ctdata0r = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    ctdata1l = ctdata0l ^ diff[0]
    ctdata1r = ctdata0r ^ diff[1]
    
    plain0l, plain0r = decrypt_simon((ctdata0l, ctdata0r), ks)
    plain1l, plain1r = decrypt_simon((ctdata1l, ctdata1r), ks_diff)
    
    delta_plain0l = plain0l ^ plain1l
    delta_plain0r = plain0r ^ plain1r
    indiffs = delta_plain0l*(2**16)+delta_plain0r
    
    del plain0l,plain0r,plain1l,plain1r,ctdata0l,ctdata0r,ctdata1l,ctdata1r,delta_plain0l,delta_plain0r
    gc.collect()
    del ks,ks_diff;gc.collect()    
    return indiffs,key_indiffs_res

def getkeydiffs(nr, n=10**4,key_diff=(0x0,0x0,0x0,0x0)):
    # print(key_diff)
    keys = np.frombuffer(urandom(8 * n), dtype=np.uint16).reshape(4, -1)
    keys_diff = np.array([keys[0] ^ key_diff[0], keys[1] ^ key_diff[1], keys[2] ^ key_diff[2], keys[3] ^ key_diff[3]],dtype=np.uint16)    
    ks = expand_key_simon(keys, nr)
    ks_diff = expand_key_simon(keys_diff, nr)
    outdiffs = np.array(ks) ^ np.array(ks_diff)
    # print(outdiffs)
    outdiffs = np.transpose(outdiffs)
    # print(outdiffs)
    # print(outdiffs.shape)
    res = []
    for outdiff in outdiffs:
        res.append("_".join([f"{hex(item)}" for item in outdiff])) 
    # print(res)
    # print(np.array(ks).shape)
    del keys,keys_diff,ks,ks_diff,outdiffs;gc.collect()
    return res



def getkeydiffs_reversed(nr, n=10**4,key_diff=(0x0,0x0,0x0,0x0)):
    # print(key_diff)
    keys = np.frombuffer(urandom(8 * n), dtype=np.uint16).reshape(4, -1)
    keys_diff = np.array([keys[0] ^ key_diff[0], keys[1] ^ key_diff[1], keys[2] ^ key_diff[2], keys[3] ^ key_diff[3]],dtype=np.uint16)    
    ks = expand_key_simon_reversed(keys, nr)
    ks_diff = expand_key_simon_reversed(keys_diff, nr)
    outdiffs = np.array(ks) ^ np.array(ks_diff)
    # print(outdiffs)
    outdiffs = np.transpose(outdiffs)
    # print(outdiffs)
    # print(outdiffs.shape)
    res = []
    for outdiff in outdiffs:
        res.append("_".join([f"{hex(item)}" for item in outdiff])) 
    # print(res)
    # print(np.array(ks).shape)
    del keys,keys_diff,ks,ks_diff,outdiffs;gc.collect()
    return res

def check():
    nr = 32
    plain = (0x6565,0x6877)
    cipher = (0xc69b,0xe9bb)
    keys = np.frombuffer(urandom(8 * 1), dtype=np.uint16).reshape(4, -1)
    # print(keys)
    keys = np.array([[0x1918],[0x1110],[0x0908],[0x0100]])
    ks = expand_key_simon(keys, nr)
    # print(ks)
    # print(np.array(ks).shape)
    c0 = encrypt_simon(plain,ks)
    assert np.all(c0 == cipher)
    
    p0 = decrypt_simon(cipher,ks)
    assert np.all(p0 == plain)
    
    keys_reversed = np.array([[36116],[11434],[10000],[59410]])
    assert np.all(expand_key_simon_reversed(keys_reversed,nr) == ks)
    
                                                                                                                                                                                                                                
check()

