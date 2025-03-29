#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <number_of_runs> <number_of_epochs>"
    exit 1
fi

if [ -z "$2" ]; then
    echo "Usage: $0 <number_of_runs> <number_of_epochs>"
    exit 1
fi

echo "Running area experiment with $1 runs and $2 epochs"

# to not run the script by mistake with diff num of epochs than usual
if [ "$2" -ne 5000 ]; then
    echo "Warning: the number of epochs is not 5000"
    echo "Are you sure you want to continue? (y/n)"
    read -r response
    if [ "$response" != "y" ]; then
        echo "Exiting..."
        exit 1
    fi
fi

n_runs="$1"
tau_values=(0.10 0.18 0.32 0.58 1.05 1.90 3.42 6.17 11.11 20.00)

# Loop over each tau and run the command n_runs times
for tau in "${tau_values[@]}"; do
    (
    for (( run=1; run<=n_runs; run++ )); do
        # echo "================================="
        echo "Running $run/$n_runs for tau=$tau"
        # echo "================================="
        python main.py -f generated/u_arrmul4.cgp -c area -p 10 -e $2 -m 0.03 -t "$tau" -d cpu --log "logs_f/area_${tau}" > /dev/null 2>&1
    done
    ) &
done

wait
echo "All done."