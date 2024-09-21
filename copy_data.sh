# Copy necessary files from /wiss

wiss="/Volumes/wiss/M203"
save="/Users/jamesruppert/code/piccolo-data/data"

echo "Sounding Level2 data"
echo " "
rsync -au $wiss/Radiosondes/level2 $save/soundings/
echo "Sounding figures"
echo " "
rsync -au $wiss/Radiosondes/figures $save/soundings/

echo "SeaSnake data"
echo " "
rsync -au $wiss/SeaSnake/ $save/SeaSnake/

echo "DSHIP data"
echo " "
rsync -au $wiss/Dship_data/data/ $save/DSHIP/

echo "Radiometer data"
echo " "
rsync -au $wiss/Radiometer_MWR-HatPro-Uni-Leipzig/Data/ $save/radiometer/

echo "Microtops data"
echo " "
rsync -au $wiss/microtops/downloaded/ $save/microtops/

echo "ISAR SeaSkinTemp data"
echo " "
rsync -au $wiss/ISAR_SeaSkinTemp/data/Processed/ $save/ISAR_seaskintemp/

echo "CloudRadar"
echo " "
rsync -au $wiss/CloudRadar/*png $save/CloudRadar/

echo "Seminar"
echo " "
rsync -au $wiss/Seminar $save/

exit