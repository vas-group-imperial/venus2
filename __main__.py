import argparse
import datetime

from venus.common.configuration import Config
from venus.verification.venus import Venus

def boolean_string(s):
    assert  s in ['False', 'True']
    return s == 'True'

def main():
    parser = argparse.ArgumentParser(description="Venus Example")
    parser.add_argument(
        "--spec", 
        type=str,
        required=True, 
        help="Vnnlib file of the specification."
    )
    parser.add_argument(
        "--net", 
        type=str, 
        required=True, 
        help="Path to the neural network in ONNX."
    )
    parser.add_argument(
        "--stability_ratio_cutoff", 
        default=None, 
        type=float, 
        help="Cutoff value of the stable ratio during the splitting procedure. Default value is 0.5."
    )
    parser.add_argument(
        "--split_proc_num", 
        default=None, 
        type=int, 
        help="Determines the number of splitting processes = 2^splitters. Default value is 0. -1 for None."
    )
    parser.add_argument(
        "--ver_proc_num", 
        default=None,
        type=int, 
        help="Number of verification processes. Default value is 1."
    )
    parser.add_argument(
        "--intra_dep_constrs", 
        default=None, 
        type=boolean_string, 
        help="Whether to include offline intra dependency contrainsts (before starting the solver) or not. Default value is True."
    )
    parser.add_argument(
        "--inter_dep_constrs", 
        default=None, 
        type=boolean_string,
        help="Whether to include offline inter dependency contrainsts (before starting the solver) or not. Default value is True."
    )
    parser.add_argument(
        "--intra_dep_cuts", 
        default=None, 
        type=boolean_string, 
        help="Whether to include online intra dependency cuts (through solver callbacks) or not. Default value is True."
    )
    parser.add_argument(
        "--inter_dep_cuts", 
        default=None, 
        type=boolean_string,
        help="Whether to include online inter dependency cuts (through solver callbacks) or not. Default value is True."
    )
    parser.add_argument(
        "--ideal_cuts", 
        default=None, 
        type=boolean_string, 
        help="Whether to include online ideal cuts (through solver callbacks) or not. Default value is True."
    )
    parser.add_argument(
        "--split_strategy", 
        choices=["node","nodeonce","input","inputnode","inputnodeonce","nodeinput","nodeonceinput","inputnodealt","inputnodeoncealt","none"], 
        default=None, 
        help="Strategies for diving the verification problem"
    )
    parser.add_argument(
        "--monitor_split",
        default=None,
        type=boolean_string,  
        help="If true branching is initiated only after the <branching_threshold> of MILP nodes is reached"
    )
    parser.add_argument(
        "--branching_depth", 
        default=None,
        type=int, 
        help="Maximum branching depth"
    )
    parser.add_argument(
        "--branching_threshold", 
        default=None,
        type=int, 
        help="MILP node thresholf before inititing branching"
    )
    parser.add_argument(
        "--timeout", 
        default=None,
        type=float, 
        help="Timeout in seconds. Default value is 3600."
    )
    parser.add_argument(
        "--logfile", 
        default="venus_log.txt",
        type=str, 
        help="Path to logging file."
    )
    parser.add_argument(
        "--sumfile", 
        default="venus_summary.txt",
        type=str, 
        help="Path to summary file."
    )
    parser.add_argument(
        "--complete",
        default=True, 
        type=boolean_string, 
        help="Complete or incomplete verification"
    )
    parser.add_argument(
        "--osip_conv",
        default=None,
        type=str,
        help="OSIP mode of operation for convolutional layers, one of 'on', 'off', 'node_once', 'node_always'"
    )
    parser.add_argument(
        "--osip_conv_nodes",
        default=None,
        type=int,
        help="Number of optimised nodes during OSIP for convolutional layers"
    )
    parser.add_argument(
        "--osip_fc",
        default=None,
        type=str, 
        help="OSIP mode of operation for fully connected layers, one of 'on', 'off', 'node_once', 'node_always'"
    )
    parser.add_argument(
        "--osip_fc_nodes",
        default=None,
        type=int,
        help="Number of optimised nodes during OSIP for fully connected layers"
    )
    parser.add_argument(
        "--osip_timelimit",
        default=None,
        type=int,
        help="Timelimit in seconds for OSIP"
    )
    parser.add_argument(
        "--relu_approximation",
        default=None, 
        type=str,
        help="Relu approximation: 'min_area' or 'identity' or 'venus' or 'parallel' or 'zero'"
    )
    parser.add_argument(
        "--console_output",
        default=None, 
        type=boolean_string,
        help="Console output switch"
    )
    
    ARGS = parser.parse_args()
    config = Config()
    config.set_user(ARGS)
    venus = Venus(nn=ARGS.net, spec=ARGS.spec, config=config)
    venus.verify()

if __name__ == "__main__":
    main()
