# /bin/csh
#
#$ -cwd -j y
#$ -o casa_m33_aca.line_staging_imaging.$JOB_ID.$TASK_ID.log
#$ -q mThC.q
#$ -l mres=16G,h_data=4G,h_vmem=4G
#$ -N m33_aca.lines
#$ -t 1-173 -tc 40
#$ -pe mthread 4

# Example script for submitting a job array to stage and image individual targets
# with the PHANGS pipeline

# Job array number corresponds to:
# 2017, 2019, 2021 ACA mapping
# 1-106  -tc 40

# 2018 ACA
# 107-127

# 2022 ACA
# 128-173

echo + `date` $JOB_NAME started on $HOSTNAME in $QUEUE with jobID=$JOB_ID and taskID=$SGE_TASK_ID
echo NSLOTS=$NSLOTS
#
set CASA = $HOME/casa-6.4.1-12-pipeline-2022.2.0.68/bin/
time \

set file_drive = scratch
set data_dir = $HOME/${file_drive}/full_aca_band6/

cd $data_dir

mkdir -p logs

# Ensure no time overlap in job start times.
# CASA can get cranky opening too many instances at once.
python3 -c "import time, random; time.sleep(random.randint(2, 120))"

set casa_log_file = $HOME/${file_drive}/full_aca_band6/logs/casa__${JOB_ID}_${SGE_TASK_ID}_`date "+%Y%m%d-%H%M%S"`.log
set casa_script = $HOME/full_aca_band6/run_casa_pipeline_lines_stage_image_permosaic_jobarray.py

# Change to use config files for a specific line product
set line_type = all

set casa_job_config_file = $HOME/full_aca_band6/keys_hydra/config_lines/line_staging_imaging.${line_type}.${SGE_TASK_ID}.jobconfig.txt

# Force CASA to use the right number of threads per cores assigned in the job array
setenv OMP_NUM_THREADS $NSLOTS

$CASA/casa --nogui --log2term --logfile $casa_log_file -c $casa_script $casa_job_config_file

echo $JOB_NAME $SGE_TASK_ID done `date`

