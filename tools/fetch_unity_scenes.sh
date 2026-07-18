#!/usr/bin/env bash
# Download the ROS2-era unity scene zips (organizers' unity_env_models Drive folder)
set -u
cd ~/vla/data/unity_scenes_ros2
GDOWN=~/vla/.venv/bin/python
declare -A IDS=(
  [arabic_room]=1_lWkeZwvnUNeiCyCH83YU8XXq5uT1A7k
  [chinese_room]=12hS_lR8bCLim19ofyWgHW6cB5kaGGssU
  [home_building_1]=1i7lBSzAy97xYO5yD6LhI5J5fsAPsPa-S
  [home_building_2]=1iMVo7021iziTPRwR-MugPfehWP1XGai0
  [hotel_room_1]=1Aol07uFQf8cz2Q2fNJTW5DkvwnVZSKjY
  [hotel_room_2]=1KoCPrhVfcDqTnk7ZJMJG4xJ570FEsX6h
  [japanese_room]=11p9ymIuBPYTL8mgRcy3HE_7LNA6NUTNN
  [livingroom_2]=1dDLWYvQ23DifW95s9TsVUzzlBXVamHXy
  [livingroom_3]=1EwkQ5cw9P-7gf_g1NHAxL38rPVgof6X_
  [livingroom_4]=1sgJ3P2cWZcUThl_3GmRwN1CTCdsV_2Wq
  [loft]=1xEJtquiqa_xiOyqYBQmh10pvzF5gIzhJ
  [office_1]=1YSTWpVdzx53VHklquexgeymh2J0UvOH9
  [office_2]=1VxTKjff-PdTJaN3IdmXpJaIbfP6dsg22
  [office_building_1]=1XDvtheDg0NCPpsnosfSoVCb1hmMdHcJK
  [office_building_2]=1VlFSObIXosY4gRkdoL86oHaBi2NqN-33
  [office_building_2_without_360_cam]=1x3XVwoQvC-VrwwH5V7aMASNtfo4zE7yH
  [studio]=1M8QoY_Ht8LKfglKM160TqFdbLKxB4Ms3
)
fails=""
for name in "${!IDS[@]}"; do
  [ -d "$name" ] && { echo "SKIP $name (exists)"; continue; }
  "$GDOWN" -m gdown "${IDS[$name]}" -O "$name.zip" >/dev/null 2>&1
  if [ -s "$name.zip" ]; then
    unzip -q -o "$name.zip" -d "$name" && echo "OK $name" || { echo "UNZIP_FAIL $name"; fails="$fails $name"; }
  else
    echo "DOWNLOAD_FAIL $name"; fails="$fails $name"
  fi
done
find . -name Model.x86_64 -exec chmod +x {} +
echo "DONE. failures:${fails:-none}"
