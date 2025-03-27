#!/bin/bash

# Usage check: provide the number of runs per tau value as the first argument
if [ -z "$1" ]; then
    echo "Usage: $0 <number_of_runs>"
    exit 1
fi

n_runs="$1"

# Hardcoded array of 10 tau values (logarithmically spaced from 0.1 to 20, rounded to 2 decimals)
tau_values=(0.10 0.18 0.32 0.58 1.05 1.90 3.42 6.17 11.11 20.00)
# tau_values=(3.42 6.17 11.11 20.00)

# Loop through each tau value
for tau in "${tau_values[@]}"; do
    # For each tau, run the command n_runs times
    for (( run=1; run<=n_runs; run++ )); do
        python main.py -f generated/u_arrmul4.cgp -c area -p 10 -e 5000 -m 0.03 -t "$tau" -d cpu --log "logs/area_${tau}"
    done
done
