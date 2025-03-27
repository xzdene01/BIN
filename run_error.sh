#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <number_of_runs>"
    exit 1
fi

n_runs="$1"
tau_values=(30 34 38 42 46 50 54 58 62 66)

# Loop over each tau and run the command n_runs times
for tau in "${tau_values[@]}"; do
    for (( run=1; run<=n_runs; run++ )); do
        python main.py -f generated/u_arrmul4.cgp -c error -p 10 -e 5000 -m 0.03 -t "$tau" -d cpu --log "logs/error_${tau}" --pretrain 5000
    done
done
