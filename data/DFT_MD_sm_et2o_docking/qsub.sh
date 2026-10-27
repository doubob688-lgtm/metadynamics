
#!/usr/bin/env bash
set -euo pipefail

# Run from the directory containing MD_* folders

for dir in MD_*/; do
    [[ -d "$dir" ]] || continue

    echo "=============================="
    echo "Processing: $dir"

    (
        cd "$dir"

        shopt -s nullglob
        pbs_files=(*.pbs)

        if (( ${#pbs_files[@]} != 1 )); then
            echo "SKIPPED: Expected exactly one PBS file"
            exit 0
        fi

        job_id=$(qsub "${pbs_files[0]}")
        echo "Submitted successfully: $job_id"
    )
done

echo "All submissions processed."

