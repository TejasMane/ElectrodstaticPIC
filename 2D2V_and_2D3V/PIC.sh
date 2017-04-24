#!/bin/bash
# Bash Menu Script Example

PS3='Please enter your choice: '
options=("2D2V" "2D3V")
select opt in "${options[@]}"
do
    case $opt in
        "2D2V")
            echo "Executing the 2D2V PIC code"
            jupyter-notebook 2D2V.ipynb
            break
            ;;
        "2D3V")
            echo "Executing the 2D3V PIC code"
            jupyter-notebook 2D3V.ipynb
            break
            ;;
        *) echo invalid option;;
    esac
done
