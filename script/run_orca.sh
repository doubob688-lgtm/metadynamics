#!/bin/bash

set -euo pipefail

# ==== User Directory Variables ==== 

#ORCA _DIR = 
PROJECT_DIR="$diazirine"
INPUT_DIR="$diazirine/data"
RUNS_DIR="$diazirine/runs"

# ==== check the command row ====
#to run the script, type ./run_orca.sh file_name input_directory_name

if [ $# -ne 2 ]; then
	echo "用法：$0 <输入文件名（不含.inp 后缀）输入文件夹名>"
	exit 1
fi

JOB_NAME="$1"
INPUT_DIRECTORY="$2"
INPUT_FILE="$INPUT_DIR/$INPUT_DIRECTORY/${JOB_NAME}.inp"

# ==== check if the file is present ====
if [ ! -f "$INPUT_FILE" ]; then
	echo "no $INPUT_FILE"
	exit 1
fi

# ==== create RUN directory ====

RUN_DIR="$RUNS_DIR/$(date +%Y-%m-%d)_${JOB_NAME}"
mkdir -p "$RUN_DIR"
cd "$RUN_DIR"

# copy the input file to the current run directory
cp "$INPUT_FILE" .

# ==== running orca ====
echo "Start running orca: $JOB_NAME"
echo "Running directory: $RUN_DIR"
orca "${JOB_NAME}.inp" > "${JOB_NAME}.out"

# 自动检查是否正常结束
if grep -q "ORCA TERMINATED NORMALLY" "${JOB_NAME}.out"; then
    echo "✅ $JOB_NAME finished normally."
else
    echo "❌ $JOB_NAME may have failed. Check ${JOB_NAME}.out"
fi

