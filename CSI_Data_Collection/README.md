# CSI data collection with Nexmon

## Install Nexmon CSI in RPi4

Please install Nexmon CSI following the main [Nexmon_CSI](https://github.com/seemoo-lab/nexmon_csi) repo. 



## Device setup and Collect Data


1. Set up the number of antenna, number of spatial streams, channel and bandwith with ***makecsiparams*** or ***mcp***

```
mcp -C 1 -N 1 -c 36/80 

```
### To collect the CSI of the channel of any particular set of transcievers, plese include the MAC address of the source and the starting bytes (0x88) to filter the frames. For example, to avoid beacon frames. 

```
mcp -C 1 -N 1 -c 36/80 -m 3c:37:86:24:52:63 -b 0x88

```
2. Make the wireless interface UP:

```
sudo ifconfig wlan0 up
```

3.

```
sudo nexutil -Iwlan0 -s500 -b -l34 -v$Output_string_from_1$

```
like this: 

```
sudo nexutil -Iwlan0 -s500 -b -l34 -vKuABEQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA==
```

4. Create a virtual interface in monitor mode

```
sudo iw dev wlan0 interface add mon0 type monitor
```

5. Make the craeted virtual interface UP

```
sudo ip link set mon0 up
```

6. Start capturing the CSI with tcpdump

```
sudo tcpdump -i wlan0 dst port 5500 -vv -w output.pcap 
```

7. Start the iperf string from the appropriate source , for downlink, it is the AP offloading to any of the STAs. However, the source can be STA too (if you want to collect the uplink CSI). 


