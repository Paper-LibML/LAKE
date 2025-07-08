#!/usr/bin/env bash

set -euo pipefail

LOCAL_DIR=""
REMOTE=""
DELAY=6

# Usage function
usage() {
    echo "Usage: $0 [-l local_dir] [-r remote] [-d delay_seconds]"
    echo
    echo "Options:"
    echo "  -l  Local directory to watch (required)"
    echo "  -r  Remote rsync destination (required)"
    echo "  -d  Delay (in seconds) after changes before syncing (default: $DELAY)"
    echo "  -h  Show this help message and exit"
    exit 1
}

# Parse options
while getopts "l:r:d:h" opt; do
    case ${opt} in
        l) LOCAL_DIR="$OPTARG" ;;
        r) REMOTE="$OPTARG" ;;
        d) DELAY="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done

if [[ -z "$LOCAL_DIR" || -z "$REMOTE" || -z "$DELAY" ]]; then
    echo "Error: options -l, and -r are required."
    usage
fi


echo "Watching $LOCAL_DIR for changes. Will sync after $DELAY seconds of inactivity."

while true; do
    # Wait for any file event
    inotifywait -r -e modify,create,delete,move "$LOCAL_DIR" >/dev/null 2>&1

    # Start debounce timer
    echo "Change detected, waiting for $DELAY seconds of inactivity..."
    while inotifywait -r -e modify,create,delete,move --timeout $DELAY "$LOCAL_DIR" >/dev/null 2>&1; do
        echo "Another change detected, resetting timer..."
    done

    # After delay with no new changes, run rsync
    echo "No changes detected for $DELAY seconds — syncing..."
    rsync -azP "$LOCAL_DIR" "$REMOTE"
    printf "Sync complete.\n\n"
done
