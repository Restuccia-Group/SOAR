INTERVAL="0.01"

search_dir=/mnt/sda1/bin-sampled-dataset
for entry in "$search_dir"/*
do
  h=$(date +%H)
  m=$(date +%M)
  s=$(date +%S)
  ms=$(date +%3N)
  now_ms=$((10#$h*3600000 + 10#$m*60000 + 10#$s*1000 + 10#$ms))
  cat $entry | nc -vu 192.168.10.2 9000;
  echo "$entry,$now_ms," >> output.txt;
  sleep $INTERVAL
done
