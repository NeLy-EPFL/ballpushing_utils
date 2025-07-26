#!/bin/bash

# This script renames all instances of major_event to first_major_event
# and major_event_time to first_major_event_time in notebooks and CSV files

echo "Updating Jupyter notebooks..."

# Update notebooks using sed (be careful with these - you may want to backup first)
find notebooks/ -name "*.ipynb" -type f -exec sed -i.bak 's/"major_event"/"first_major_event"/g' {} \;
find notebooks/ -name "*.ipynb" -type f -exec sed -i.bak 's/"major_event_time"/"first_major_event_time"/g' {} \;
find notebooks/ -name "*.ipynb" -type f -exec sed -i.bak 's/major_event_time/first_major_event_time/g' {} \;
find notebooks/ -name "*.ipynb" -type f -exec sed -i.bak 's/major_event\b/first_major_event/g' {} \;

echo "Updating CSV files..."

# Update CSV headers (backup originals first)
find . -name "*.csv" -type f -exec cp {} {}.backup \;
find . -name "*.csv" -type f -exec sed -i.bak 's/major_event_time/first_major_event_time/g' {} \;
find . -name "*.csv" -type f -exec sed -i.bak 's/major_event,/first_major_event,/g' {} \;
find . -name "*.csv" -type f -exec sed -i.bak 's/"major_event"/"first_major_event"/g' {} \;

echo "Done! Backup files created with .backup extension"
echo "Please review the changes before committing"
