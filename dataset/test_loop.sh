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

for entry in "$search_dir"/*
do
  now_ms=$(date +%Y/%M/%d-%T.%N)
  cat $entry | nc -uvc $add $port;
  echo "$entry,$now_ms" >> $name;
  sleep $interval
done
