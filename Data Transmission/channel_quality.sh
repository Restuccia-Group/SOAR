while getopts o: flag
do 
	case "${flag}" in
		o) output=${OPTARG};;
	esac
done


echo "Folder: "$output
echo "timestamp,mac_address,tx_bytes,rx_bytes,signal,signal_avg,tx_bitrate,rx_bitrate" > $output
while :; do
	echo -n $(date +%Y/%m/%d-%T.%N)",">>$output
	iw dev wlan0 station get 08:02:8e:99:9f:49 | awk '/08:02:8e:99:9f:49/{st0=$2} /tx bytes:/{st5=$3} /rx bytes:/{st1=$3} /signal:/{st2=$2$3$4$5$6} /signal avg:/{st3=$3$4$5$6} /tx bitrate:/{st4=$3$4}
	/rx bitrate:/{print st0","st5","st1","st2","st3","st4","$3"_"$4"_"$5"_"$6"_"$7"_"$8"_"$9"_"$10"_"%11}' >> $output
	
	echo -n $(date +%Y/%m/%d-%T.%N)",">>$output
	iw dev wlan0 station get bc:a5:11:1f:f7:3f | awk '/bc:a5:11:1f:f7:3f/{st0=$2} /tx bytes:/{st5=$3} /rx bytes:/{st1=$3} /signal:/{st2=$2$3$4$5$6} /signal avg:/{st3=$3$4$5$6} /tx bitrate:/{st4=$3$4}
	/rx bitrate:/{print st0","st5","st1","st2","st3","st4","$3"_"$4"_"$5"_"$6"_"$7"_"$8"_"$9"_"$10"_"%11}' >> $output
               
done

