import train_nets as tn
from time import time
import sys
import os
import argparse

def pad_and_split_hex_string(hex_str, target_length):  
    """  
    Pad a hexadecimal string to a target length and split it into an array of 4-digit hex numbers.  
      
    :param hex_str: The hexadecimal string to pad and split (should start with "0x").  
    :param target_length: The total length of the hexadecimal string after padding (without the "0x" prefix).  
    :return: A list of integers representing the 4-digit hex numbers.  
    """  
    if not hex_str.startswith('0x'):  
        raise ValueError("The input string is not a valid hexadecimal string (it does not start with '0x').")  
      
    # Remove the '0x' prefix  
    hex_value = hex_str[2:]  
      
    # Calculate the number of zeros to pad  
    pad_length = target_length - len(hex_value)  
    if pad_length < 0:  
        raise ValueError("The target length is smaller than the length of the hexadecimal value.")  
      
    # Pad the hexadecimal value with leading zeros  
    padded_hex_value = '0' * pad_length + hex_value  
      
    # Split the padded hexadecimal value into 4-digit chunks  
    hex_chunks = [padded_hex_value[i:i+4] for i in range(0, len(padded_hex_value), 4)]  
      
    # Convert the hex chunks to integers and return the list  
    int_values = [int(chunk, 16) for chunk in hex_chunks]  
      
    return tuple(int_values)  
  
if __name__=='__main__':

    
    parser = argparse.ArgumentParser()
    
    
    # input file location
    parser.add_argument("--diff",type=str,default="0x0")
    parser.add_argument("--key_diff",type=str,default="0x0")
    parser.add_argument("--GPU_device",type=str,default="0")
    parser.add_argument("--curr_time_str",type=str,default="")
    parser.add_argument("--batch_size",type=int,default=30000)
    parser.add_argument("--s_groups",type=int,default=8)
    parser.add_argument("--epochs",type=int,default=120)
    parser.add_argument("--num_rounds",type=int,default=9)
    parser.add_argument("--depth_value",type=int,default=5)
    parser.add_argument("--file_name",type=str,default="default")
    parser.add_argument("--lr_choice",type=str,default="gohr")
    parser.add_argument("--d0",type=int,default=512)
    parser.add_argument("--d1",type=int,default=128)
    parser.add_argument("--d2",type=int,default=128)
    parser.add_argument("--ks_value_1",type=int,default=1)
    parser.add_argument("--ks_value_2",type=int,default=3)
    parser.add_argument("--ks_value_3",type=int,default=3)
    parser.add_argument("--dropout_value",type=float,default=0.5)
    parser.add_argument("--machine_name",type=str,default="vgc")
    parser.add_argument("--num_filters_1",type=int,default=64)
    parser.add_argument("--num_filters_2",type=int,default=64)
    parser.add_argument("--num_filters_3",type=int,default=64)
    parser.add_argument("--reg_param",type=float,default=10**-5)
    parser.add_argument("--high_lr",type=float,default=0.003)
    parser.add_argument("--low_lr",type=float,default=0.0001)
    parser.add_argument("--lr_epoch",type=int,default=30)
    parser.add_argument("--se_ratio",type=int,default=16)
    args = parser.parse_args()
    
    
    # diff = pad_and_split_hex_string(args.diff,8)
    # key_diff = pad_and_split_hex_string(args.key_diff,16)
    # wdir = f"./{hex(diff[0])}_{hex(diff[1])}/{hex(key_diff[0])}_{hex(key_diff[1])}_{hex(key_diff[2])}_{hex(key_diff[3])}/"
    
    diffs = args.diff.split("_")
    diff0 = pad_and_split_hex_string(diffs[0],8)
    diff1 = pad_and_split_hex_string(diffs[1],8)
    diff = (diff0,diff1)
    
    key_diffs = args.key_diff.split("_")
    key_diff0 = pad_and_split_hex_string(key_diffs[0],16)
    key_diff1 = pad_and_split_hex_string(key_diffs[1],16)
    key_diff = (key_diff0,key_diff1)    
    wdir = f"./{hex(diff[0][0])}_{hex(diff[0][1])}_{hex(diff[1][0])}_{hex(diff[1][1])}/{hex(key_diff[0][0])}_{hex(key_diff[0][1])}_{hex(key_diff[0][2])}_{hex(key_diff[0][3])}_{hex(key_diff[1][0])}_{hex(key_diff[1][1])}_{hex(key_diff[1][2])}_{hex(key_diff[1][3])}/"

    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_device
    import tensorflow as tf
    config = tf.compat.v1.ConfigProto()  
    config.gpu_options.allow_growth = True  
    session = tf.compat.v1.Session(config=config)
    filename = args.file_name
    
    if args.curr_time_str == "":
        import datetime
        curr_time = datetime.datetime.now()
        curr_time_str = curr_time.strftime("%Y-%m-%d-%H-%M-%S")
    else:
        curr_time_str = args.curr_time_str


    start = time()
    tn.train_distinguisher(diff=diff,key_diff=key_diff,wdir=wdir,curr_time_str=curr_time_str,num_epochs=args.epochs,num_rounds = args.num_rounds,depth=args.depth_value,filename=args.file_name,s_groups=args.s_groups,bs=args.batch_size,lr_choice=args.lr_choice,d0=args.d0,d1=args.d1,d2=args.d2,ks_value_1=args.ks_value_1,ks_value_2=args.ks_value_2,ks_value_3=args.ks_value_3,machine_name=args.machine_name,dropout_value=args.dropout_value,num_filters_1=args.num_filters_1,num_filters_2=args.num_filters_2,num_filters_3=args.num_filters_3,reg_param=args.reg_param,high_lr=args.high_lr,low_lr=args.low_lr,lr_epoch=args.lr_epoch,se_ratio=args.se_ratio)
    end = time()
    print("time is: "+str(end-start)+" s")
