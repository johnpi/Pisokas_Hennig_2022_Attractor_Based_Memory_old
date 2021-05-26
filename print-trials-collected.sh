#!/bin/bash

for net in NMDA EC_LV_1; 
do
  echo 
  echo "${net}"
  echo "--------"
  for N in 128 256 512 1024 2048 4096 8192;
  do 
    for n in 0.001 0.002 0.003 0.004 0.005 0.006 0.007 0.008 0.009; 
    do 
      x=$(cat "${1}" | grep _${N} | grep ${net} | grep ${n} | sed 's/.*(//' | sed 's/,).*//'); 
      sum=0; 
      for num in $(echo $x); 
      do 
        sum=$((sum + $num)); 
      done; 
      echo "N=${N} noise=${n} trials=${sum}"; 
    done;
    echo 
  done;
done
