#!/bin/bash

# Array of tau/t values
tau_values=(0.1 0.2 1 2 10 20)

# Loop through each tau/t value and run the command
for tau in "${tau_values[@]}"; do
    python main.py -f generated/u_arrmul4.cgp -c area -p 10 -e 10000 -m 0.03 -t "$tau" --finetune 1000 -d cuda --log "logs/_${tau}_"
done