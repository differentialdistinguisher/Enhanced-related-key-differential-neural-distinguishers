# Enhanced-related-key-differential-neural-distinguishers-for-simon-and-simeck-block-ciphers



## Basci 
The basic related-key differential neural distinguisher for SIMON and SIMECK
## Enhanced  
The enhanced related-key differential neural distinguisher for SIMON and SIMECK

## An example to run our code
Training a 13-round basic related-key differential neural distinguisher for SIMON3264 with plaintext differce (0x0, 0x2004) and key differce (0x0, 0x0, 0x0, 0x2004) : 

```bash
cd ./Basic/Simon3264

python train_rounds.py --diff="0x00002004" --key_diff="0x0000000000002004" --num_rounds=13
```
- diff: The binary string of the plaintext difference.

- key_diff: The binary string of the mast key difference.

- num_rounds: The number of rounds to train.
  
- More parameters can be found in the file "train_rounds.py".

## References

```bash
[1] Lu J, Liu G, Sun B, et al. Improved (related-key) differential-based neural distinguishers for SIMON and SIMECK block ciphers[J]. The Computer Journal, 2024, 67(2): 537-547.https://github.com/JIN-smile/Improved-Related-key-Differential-based-Neural-Distinguishers

```
