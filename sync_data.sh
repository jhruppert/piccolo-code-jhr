# Copy necessary files from /wiss

wiss="/Volumes/wiss/M203"
save="/Users/jamesruppert/code/piccolo-data/data"

echo " "
echo "BOWTIE data paper notes"
echo " "
rsync -auv $wiss/bowtie_dataset $save/

echo " "
echo "Seminar"
echo " "
rsync -auv $wiss/Seminar $save/

echo " "
echo "PICCOLO science doc"
echo " "
rsync -auv "$wiss/SEA-POL/PICCOLO Science.docx" $save/

echo " "
echo "Sounding Level2 data"
echo " "
rsync -auv $wiss/Radiosondes/level2 $save/soundings/
echo "Sounding figures"
echo " "
rsync -auv $wiss/Radiosondes/figures $save/soundings/

echo " "
echo "SeaSnake data"
echo " "
# rsync -auv $wiss/SeaSnake/ $save/SeaSnake/
rsync -auv $wiss/SeaSnake/seaSnakeData/ $save/SeaSnake/seaSnakeData/
rsync -auv $wiss/SeaSnake/seaSnakeDataClean/ $save/SeaSnake/seaSnakeDataClean/

echo " "
echo "DSHIP data"
echo " "
rsync -auv $wiss/Dship_data/data/ $save/DSHIP/

echo " "
echo "Radiometer data"
echo " "
rsync -auv $wiss/Radiometer_MWR-HatPro-Uni-Leipzig/Data/ $save/radiometer/

echo " "
echo "Microtops data"
echo " "
rsync -auv $wiss/microtops/downloaded/ $save/microtops/

echo " "
echo "ISAR SeaSkinTemp data"
echo " "
rsync -auv $wiss/ISAR_SeaSkinTemp/data/Processed/ $save/ISAR_seaskintemp/

echo " "
echo "CloudRadar"
echo " "
rsync -auv $wiss/CloudRadar/*png $save/CloudRadar/

echo " "
echo " "
echo "Current storage size: " `du -sh .`
echo " "
echo " "

exit