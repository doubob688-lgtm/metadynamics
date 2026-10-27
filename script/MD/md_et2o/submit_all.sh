#!/bin/bash
# Submit every generated method-structure PBS task.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/../../.." && pwd)"
run_root="${diazirine:-$project_root}/run/MD/md_et2o"

for pbs_file in "$run_root"/*/run_*/run_md.pbs; do
    task_dir="$(dirname -- "$pbs_file")"
    (
        cd "$task_dir"
        qsub run_md.pbs
    )
done
