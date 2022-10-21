iw phy phy0 set antenna 0x7
hostapd -B hostapd_ch_149_bw80.conf
ifconfig wlan0 192.168.10.1
iw dev wlan0 set bitrates vht-mcs-5 3:0-9
iw phy0 info
