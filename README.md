# Simulator of Traffic Splitting for Tor
Simulator to apply splitting strategies as defense for WFP attacks.
It simulates the effects of sending packets over different routes characterized by their latency. 
Simulator used in the paper https://dl.acm.org/doi/10.1145/3372297.3423351.
Please refer to the paper for more information regarding the available splitting strategies


## Requirements
File with multi-path circuit latencies. Look at the ```circuits_latencies_new.txt``` for knowing how latencies are related to multiple circuits. 

Dataset in the wang-format. I.e. a folder containing one file per instance, in which timestamps and directions are described per packet. It can also contain a third column of packet size.

Instances should be named in wang-format as for example  0-0.cell is first instance of the page labeled as 0 and 23-5.cell is the 6th instance of page of label 23
## Usage
For getting all options and modes, run:

```bash
python simulator.py -h 
```

## Examples
For simulating a weighted random (WR) and BWR scheduler for three paths with the dataset located in ```dataset/``` and putting the output traces on ```outfolder/``` (must be created before)
```bash
##For WR m=3
python simulator.py -p dataset/ -m 3 -o outfolder -s weighted_random -i circuits_latencies_new.txt

##For BWR m=5
python simulator.py -p dataset/ -m 5 -o outfolder -s batched_weighted_random -r 50,70 -a 1,1

```

## Notes

Instance files **must** have the extension ```.cell```. The simulator will produce on the ```oufolder``` **m** (number of paths) files with extension ```_split_i.cell``` (for the i-th path), and a file with the extension ```_join_.cell``` for the merged trace but after the multi-path effect


