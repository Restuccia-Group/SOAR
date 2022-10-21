# MU-MIMO-Drone-Offloading

In this project, we investigate the various challenges of Data offloading from drone to the ground stations exploiting the downlink beamforming with multiple receive & sending antenna as well as streams. We intend to address the challenges with State-Of-The-Art Deep Learning techniques by defining the required offloading stratagies. 

For the  drone offloading task, we consider IEEE 802.11ac / 11 ax Multi User Multiple Input Multiple Output (MU-MIMO) system where Drones will be offloading various tasks like image detection, classification or segmentation to the  ground stations. Thus, the Drone is considered as the Access Point (AP) whereas the ground nodes are the Stations (STA) of the MU-MIMO system. 

We devide this whole system into several sections as follows: 

##### **1. Device Configurations**
##### **2. Installing Required Packages**
##### **3. Setup the MU-MIMO**
##### **4. Accessing and Configuring the Network Remotely**
##### **5. Data Offloading** 

## 1. Device Confifurations 

For the flexibility of the different configurations of MU-MIMO we use Netgear Nighthawk R7800 router as both AP and STAs which supports IEEE 802.11ac. Our first step would be to flash the routers with OpenWrt which allows us to exploit tools like iw, ifconfig, hostapd etc. We will go with the TFTP flashing which is more easier and convenient than that of the other exixting flashing processes. Please find the openwrt-22.03.2 image (which we will flash) for the R7800 in Device_Configuration folder.

### Prerequisites for TFTP flashing

1. A TFTP client for our computer. In Ubuntu, install it with 
```
sudo apt install tftp 
```
2. Plug the ethernet cable to the computer and LAN port 4 of the router (actually any LAN port should be ok). Make the IP address of the connected ethernet port of the PC as following: <br />
*IP address: anything in 192.168.1.2 - 192.168.1.253 <br />
Subnet mask: 255.255.255.0  <br />
Default Gateway: 192.168.1.1 <br />*

### Flash with OpenWrt

1. Turn off the power, push and hold the reset button (in a hole on backside) with a pin
2. While still holding the reset button, turn on the power, and the power led starts flashing white (after it first flashes orange for a while)
3. Release the reset button after the power led flashes white (for at least 10 times), immediately execute the tftp command on your computer. 
```
tftp -i [router IP] put [firmware filename].[file format]
```
For our case, it would be: 
``` 
tftp -i 192.168.1.1 put openwrt-22.03.2_r7800-squashfs.img
```
4. The power led will stop flashing if succeeded in transferring the image, and the router reboots rather quickly with the new firmware.
5. After it boots, it should have the default IP of OpenWrt which is "192.168.1.1". Try to ping "192.168.1.1" from the host PC. 
6. If the pinging works fine, go ahead with  ``` ssh root@192.168.1.1``` and we should be logged in as by default, no password is set.   


## 2. Installing Required Packages
1. For installing packages, we have two choices, (i) download the required packages in "ipk" format in host PC, scp the file to the R7800, and install with opkg. \newline
*As we can see from the image name that our OpenWrt vesion is 22.03.2, we should look for the ipk releases for the 22.03.2 version*

For our case, all the necessary packages can be found [here](https://archive.openwrt.org/releases/22.03.2/packages/arm_cortex-a15_neon-vfpv4/packages/)
  
