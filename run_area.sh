#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <number_of_runs>"
    exit 1
fi

n_runs="$1"
tau_values=(0.10 0.18 0.32 0.58 1.05 1.90 3.42 6.17 11.11 20.00)

# Loop over each tau and run the command n_runs times
for tau in "${tau_values[@]}"; do
    for (( run=1; run<=n_runs; run++ )); do
        echo "================================="
        echo "Running $run/$n_runs for tau=$tau"
        echo "================================="
        python main.py -f generated/u_arrmul4.cgp -c area -p 10 -e 2000 -m 0.03 -t "$tau" -d cpu --log "logs/area_${tau}"
    done
done
