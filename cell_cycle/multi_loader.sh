cd /home/shenghuahe/segmentation_models/cell_cycle
python2 job_parser.py 'multi_GPU.sh'
for i in $(seq 0 3)
do
   sh job_folder/job_$i.sh&
   sleep 30s &
done
wait
