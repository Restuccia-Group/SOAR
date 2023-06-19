while getopts d:a:p:n:i: flag
do 
    case "${flag}" in
      d) search_dir=${OPTARG};;
	    a) add=${OPTARG};;
	    p) port=${OPTARG};;
	    n) name=${OPTARG};;
	    i) interval=${OPTARG};;
    esac
done

echo "search_dir: $search_dir" 
echo "address: $add"
echo "port: $port"
echo "name: $name"
echo "interval: $interval"

while :; do
 for entry in "$search_dir"/*
  do
    sleep $interval
    echo "$entry,$(date +%Y/%m/%d-%T.%N)" >> $name;
    cat $entry | ncat -vu $add $port;
    echo "$entry,$(date +%Y/%m/%d-%T.%N)" >> $name;
    echo "NEW_STREAM" | ncat -tv $add $port;
  done
done
