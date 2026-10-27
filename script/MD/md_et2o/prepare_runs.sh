#!/bin/bash
# Generate one ORCA MD task directory for every method-structure combination.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/../../.." && pwd)"
run_root="${diazirine:-$project_root}/run/MD/md_et2o"
methods_file="$project_root/data/md_et2o/method_inputs/methods.txt"
xyz_dir="$project_root/data/md_et2o/xyz_inputs"

method_index=0
while IFS= read -r method; do
    method_index=$((method_index + 1))
    method_name="$(sed -E 's/\(([^)]*)\)/_\1/g; s/[^[:alnum:].+-]+/_/g; s/_+/_/g; s/^_|_$//g' <<< "$method")"
    method_dir="$run_root/$method_name"
    mkdir -p "$method_dir"
#md_et2o
    xyz_index=0
    for xyz_source in "$xyz_dir"/*.xyz; do
        xyz_index=$((xyz_index + 1))
        xyz_file="$(basename -- "$xyz_source")"
        xyz_name="${xyz_file%.xyz}"
        task_dir="$method_dir/run_$xyz_name"
        hpc_log_dir="$task_dir/hpc_log"
        job_name="md_${method_index}_${xyz_index}"

        mkdir -p "$hpc_log_dir"
        cp -- "$xyz_source" "$task_dir/$xyz_file"
        sed \
            -e "s|__METHOD__|$method|g" \
            -e "s|__XYZ_FILE__|$xyz_file|g" \
            "$script_dir/run_md.inp" > "$task_dir/run_md.inp"
        sed \
            -e "s|__JOB_NAME__|$job_name|g" \
            -e "s|__HPC_LOG_DIR__|$hpc_log_dir|g" \
            "$script_dir/run_md.pbs" > "$task_dir/run_md.pbs"
    done
done < "$methods_file"
