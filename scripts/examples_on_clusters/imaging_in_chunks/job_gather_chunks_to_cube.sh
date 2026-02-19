# /bin/csh
#
# Example script for submitting a job array to stage and image individual targets
# with the PHANGS pipeline

#$ -cwd -j y
#$ -o casa_gather_chunks.$JOB_ID.$TASK_ID.log
#$ -q mThC.q
#$ -pe mthread 4
#$ -l mres=16G,h_data=4G,h_vmem=4G
#$ -N casa_gather_chunks

echo + `date` $JOB_NAME started on $HOSTNAME in $QUEUE with jobID=$JOB_ID
echo NSLOTS=$NSLOTS
#
set CASA = $HOME/casa-6.4.1-12-pipeline-2022.2.0.68/bin/

set file_drive = scratch
set data_dir = $HOME/${file_drive}/to_data/

cd $data_dir

# Ensure no time overlap in job start times.
# CASA can get cranky opening too many instances at once.
python3 -c "import time, random; time.sleep(random.randint(2, 120))"

set casa_script = run_casa_gather_chunks.py

# Force CASA to use the right number of threads per cores assigned in the job array
setenv OMP_NUM_THREADS $NSLOTS

$CASA/casa --nogui --log2term -c $casa_script $SGE_TASK_ID

echo $JOB_NAME $SGE_TASK_ID done `date`

