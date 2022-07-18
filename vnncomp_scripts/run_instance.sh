#!/bin/bash
# example run_benchmark.sh script for VNNCOMP for nnenum 
# six arguments, first is "v1", second is a benchmark category itentifier string such as "acasxu", third is path to the .onnx file, fourth is path to .vnnlib file, fifth is a path to the results file, and sixth is a timeout in seconds.
# Stanley Bak, Feb 2021

TOOL_NAME=venus2
VERSION_STRING=v1

# check arguments
if [ "$1" != ${VERSION_STRING} ]; then
	echo "Expected first argument (version string) '$VERSION_STRING', got '$1'"
	exit 1
fi

DIR=$(dirname $(dirname $(realpath $0)))
export GUROBI_HOME="$DIR/gurobi912/linux64"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:$DIR/gurobi912/linux64/lib"
export PATH="${PATH}:${GUROBI_HOME}/bin"

CATEGORY=$2
ONNX_FILE=$3
VNNLIB_FILE=$4
RESULTS_FILE=$5
TIMEOUT=$6

echo "Running $TOOL_NAME on benchmark instance in category '$CATEGORY' with onnx file '$ONNX_FILE', vnnlib file '$VNNLIB_FILE', results file $RESULTS_FILE, and timeout $TIMEOUT"

# setup environment variable for tool (doing it earlier won't be persistent with docker)"
DIR=$(dirname $(dirname $(realpath $0)))
export PYTHONPATH="$PYTHONPATH:$DIR/"

# run the tool to produce the results file
python3 . --net "$ONNX_FILE" --property "$VNNLIB_FILE" --timeout "$TIMEOUT" --sumfile "$RESULTS_FILE" --benchmark "$CATEGORY"
