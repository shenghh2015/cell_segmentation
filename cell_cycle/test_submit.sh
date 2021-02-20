export LSF_DOCKER_VOLUMES='/scratch1/fs1/anastasio/Data_FDA_Breast/phase_cells:/data'
export LSF_DOCKER_NETWORK=host
export LSF_DOCKER_IPC=host 
export LSF_DOCKER_SHM_SIZE=40G
bsub -G compute-anastasio -n 1 -R 'span[ptile=1] select[mem>40000] rusage[mem=40GB]' -q general -a 'docker(shenghh2020/tf_gpu_py3.5:2.0)' -gpu "num=4" -o ~/phase_cells/logs/1GPU_$RANDOM /bin/bash ~/segmentation_models/cell_cycle/test_life/run_cyc2.sh