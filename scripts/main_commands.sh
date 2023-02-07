#!/bin/bash
for value in {1..4}
do
    echo "Round" $value >> python_output.txt
    python edge_server_perform_training.py  >> python_output.txt && python edge_server_calculate_acc.py >> python_output.txt && python edge_server_generate_validation.py >> python_output.txt && python edge_server_deploy.py >>  python_output.txt

done