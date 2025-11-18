#!/bin/bash
echo "Stopping drowsiness detection"
PID=$(pgrep -f "python main.py")
if [ -n "$PID" ]; then
    echo "Stopping process with PID $PID"
    kill $PID
    wait $PID
    echo "Drowsiness detection stopped successfully"
else
    echo "No drowsiness detection process found"
fi