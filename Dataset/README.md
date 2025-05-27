# Data log and transmission

## Requirements for the OpenWRT of the router
The Busy Box version of the packages that we use to send and monitor the packets is very limited for the experiments we will perform. Therefore, we suggest to install ``tcpdump`` and ``ncat`` from ``opkg`` using the following commands. 
```
opkg update
opkg install tcpdump
opkg install ncat
opkg install coreutils-install
```

UDP data transmission
--------------------
After the AP and the stations are working with the MU-MIMO configuration (i.e. we can ping between routers).We will copy the datasets of binary images and the ``test_loop.sh`` file to the USB driver of the AP.
```
scp -r ./bin-sampled-dataset root@192.168.1.1:/mnt/sda1
scp -r ./bin-sampled-dataset-person root@192.168.1.1:/mnt/sda1
scp test_loop.sh root@192.168.1.1:/mnt/sda1
```

We use ``ncat`` to send the files from the dataset through UDP. We initate the server side indicating the transport procool (UDP in our case), we also enable ``ncat`` verbose to be able to store the received data and we set zero waiting time to receive streams without extra delays. We suggest to perform the following command in the USB driver location to dump the received info to the USB and not in the router's memory. 
```
root@OpenWrt:/mnt/sda1# netcat -luvz -p 9000 > 001-bin-images.bin
```
Please change the name of the output binary file everytime there is a new experiment. Also, this command runs on the STAtions, because they received the images from the AP.

To monitor the received data, we use ``tcpdump`` at the server side of ``ncat``. We open another terminal at the STAtion to store the packets information coming from the specific interface. We have been using the ``wlan0`` interface, but this could change depending on the configuration performed. We use the following command to store the ```pcap`` file generated from ``tcp dump```. This step will be repeated in both devices (STAs and AP).

```
root@OpenWrt:/mnt/sda1# tcpdump -n -i wlan0 -s 65535 -w 001-sta-1-packets_bin.pcap
```
Since we are storing data, please execute the command above on the USB driver location. 

On the client side of the ``ncat`` command we will use the ``test_loop.sh``. This file receives as arguments ``-d`` (directory where the binary images are located), ``-a`` (the local area address of the station where the data will be sent), ``-p`` (port that will be used), ``-n`` (the name and extension of the output file),``-i``(the inteval in seconds of the sending data rate). The output file stores the name of each image sent as well as the timestamp of when each binary image is sent. An execution example of the command where we use the ``bin-sampled-dataset`` to send images to ``192.168.10.2`` through the port ``9000`` every ``1`` second and we store the seding time stamps in a file called ``001-send-timestamps.txt``:
```
root@OpenWrt:/mnt/sda1# ./test_loop.sh -d "/mnt/sda1/bin-sampled-dataset" -a "192.168.10.2" -p 9000 -n 001-send-timestamps.txt -i 0
```
Tip: Sometimes the test_loop requires bash execution permissions. In that case, please use: ``chmod +x test_loop.sh``

In the end we should have three files per experiment:
- pcap file: stores the UDP packets information 
- timestamps sending file: stores when the images where sent (to get the delay)
- binary received file: stores the packets content (to reconstruct the images that were sent)
