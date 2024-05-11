import numpy as np
from os import urandom
import gc

def WORD_SIZE():
    return(24)

MASK_VAL = 2 ** WORD_SIZE() - 1


const_simeck = [0xfffffd, 0xfffffd, 0xfffffd, 0xfffffd, 0xfffffd, 0xfffffc, 0xfffffc, 0xfffffc, 0xfffffd, 0xfffffd, 0xfffffc, 0xfffffd, 0xfffffd, 0xfffffd, 0xfffffc, 0xfffffd, 0xfffffc, 0xfffffd, 0xfffffc, 0xfffffc, 0xfffffc, 0xfffffc, 0xfffffd, 0xfffffc, 0xfffffc, 0xfffffd, 0xfffffc, 0xfffffd, 0xfffffd, 0xfffffc, 0xfffffc, 0xfffffd, 0xfffffd, 0xfffffd, 0xfffffd, 0xfffffd]


def rol(x,k):
    return(((x << k) & MASK_VAL) | (x >> (WORD_SIZE() - k))) 


def enc_one_round_simeck(p, k):
    c1 = p[0] 
    c0 = (rol(p[0], 5) & rol(p[0],0)) ^ rol(p[0],1) ^ p[1] ^ k
    return(c0,c1)


def expand_key_simeck(k, t):
    ks = [0 for i in range(t)]
    ks_tmp = [0,0,0,0]
    ks_tmp[0] = k[3]
    ks_tmp[1] = k[2]
    ks_tmp[2] = k[1]
    ks_tmp[3] = k[0]
    ks[0] = ks_tmp[0]
    for i in range(1, t):
        ks[i] = ks_tmp[1]
        tmp = (rol(ks_tmp[1], 5) & rol(ks_tmp[1], 0)) ^ rol(ks_tmp[1], 1) ^ ks[i-1] ^ const_simeck[i-1]
        ks_tmp[1] = ks_tmp[2]
        ks_tmp[2] = ks_tmp[3]
        ks_tmp[3] = tmp
    return(ks)


def encrypt_simeck(p, ks):
    x, y = p[0], p[1]
    for k in ks:
        x,y = enc_one_round_simeck((x,y), k)
    return(x, y)


def convert_to_binary(arr,s_groups=1):
  X = np.zeros((8 * WORD_SIZE() * s_groups,len(arr[0])),dtype=np.uint8) 
  for i in range(8 * WORD_SIZE() * s_groups):
    index = i // WORD_SIZE() 
    offset = WORD_SIZE() - (i % WORD_SIZE()) - 1
    X[i] = (arr[index] >> offset) & 1
  X = X.transpose() 
  return(X)


# def make_train_data(n, nr, diff=(0x0,0x40),s_groups=1):
def make_train_data(n, nr, diff=(0x0,0x40),key_diff=(0x0,0x0,0x0,0x0),s_groups=1):
    Y = np.frombuffer(urandom(n), dtype=np.uint8)
    Y = Y & 1
    # keys = np.frombuffer(urandom(12 * n), dtype=np.uint32).reshape(4, -1)
    keys = np.frombuffer(urandom(16 * n), dtype=np.uint32) & MASK_VAL
    keys = keys.reshape(4, -1)
    # print([hex(key) for key in keys[0]])
    # print(keys.shape)
    # print(keys)
    keys_diff = np.array([keys[0] ^ key_diff[0], keys[1] ^ key_diff[1], keys[2] ^ key_diff[2], keys[3] ^ key_diff[3]],dtype=np.uint32)
    # print([hex(key) for key in keys_diff[0]])
    
    num_rand_samples = np.sum(Y == 0)
    ks = expand_key_simeck(keys, nr)
    ks_diff = expand_key_simeck(keys_diff, nr)
    X_result = []
    del keys,keys_diff;gc.collect()
    
    
    for i in range(s_groups):
        plain0l = np.frombuffer(urandom(4 * n), dtype=np.uint32) & MASK_VAL
        plain0r = np.frombuffer(urandom(4 * n), dtype=np.uint32) & MASK_VAL
        plain1l = plain0l ^ diff[0]
        plain1r = plain0r ^ diff[1]
        plain1l[Y == 0] = np.frombuffer(urandom(4 * num_rand_samples), dtype=np.uint32) & MASK_VAL
        plain1r[Y == 0] = np.frombuffer(urandom(4 * num_rand_samples), dtype=np.uint32) & MASK_VAL
        ctdata0l, ctdata0r = encrypt_simeck((plain0l, plain0r), ks)
        ctdata1l, ctdata1r = encrypt_simeck((plain1l, plain1r), ks_diff)
        
        delta_ctdata0l = ctdata0l ^ ctdata1l
        delta_ctdata0r = ctdata0r ^ ctdata1r
        
        # delta_ctdata0rr = ctdata0l ^ ctdata1l ^ ctdata0r ^ ctdata1r
        
        # delta_ctdata0 = ctdata0l ^ ctdata0r
        # delta_ctdata1 = ctdata1l ^ ctdata1r

        secondLast_ctdata0r = rol(ctdata0r, 5) & rol(ctdata0r, 0) ^ rol(ctdata0r, 1) ^ ctdata0l
        secondLast_ctdata1r = rol(ctdata1r, 5) & rol(ctdata1r, 0) ^ rol(ctdata1r, 1) ^ ctdata1l
        
 
        delta_secondLast_ctdata0r =  secondLast_ctdata0r ^ secondLast_ctdata1r
        
        
        thirdLast_ctdata0r = ctdata0r ^ rol(secondLast_ctdata0r,5) & rol(secondLast_ctdata0r,0) ^ rol(secondLast_ctdata0r,1)
        thirdLast_ctdata1r = ctdata1r ^ rol(secondLast_ctdata1r,5) & rol(secondLast_ctdata1r,0) ^ rol(secondLast_ctdata1r,1)
        
        
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
        del plain0l,plain0r,plain1l,plain1r,ctdata0l,ctdata0r,ctdata1l,ctdata1r,delta_ctdata0l,delta_ctdata0r,secondLast_ctdata0r,secondLast_ctdata1r,delta_thirdLast_ctdata0r,delta_secondLast_ctdata0r
        gc.collect()
    
    X= convert_to_binary(X_result,s_groups=s_groups)
    #X = np.tile(X,s_groups)
    del ks,ks_diff,X_result;gc.collect()
    return (X, Y)


def check():
    nr = 36
    plain = (0x726963,0x20646e)
    cipher = (0xf3cf25,0xe33b36)
    # keys = np.frombuffer(urandom(8 * 1), dtype=np.uint32).reshape(4, -1)
    # print(keys)
    keys = np.array([[0x1a1918],[0x121110],[0x0a0908],[0x020100]])
    ks = expand_key_simeck(keys, nr)
    # print(ks)
    # print(np.array(ks).shape)
    c0 = encrypt_simeck(plain,ks)
    assert np.all(c0 == cipher)
    
    # p0 = (cipher,ks)
    # assert np.all(p0 == plain)
    
    # keys_reversed = np.array([[36116],[11434],[10000],[59410]])
    # assert np.all(expand_key_simeck_reversed(keys_reversed,nr) == ks)
    
     
if __name__ == '__main__':                                                                                                                                                                                                                        
    check()
    make_train_data(10,5)