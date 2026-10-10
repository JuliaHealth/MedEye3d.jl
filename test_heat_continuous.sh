#!/bin/bash
# W key to enter edit mode
echo "KEY W" > /tmp/test_inject.txt
sleep 0.5

# M key to hide old lesions
echo "KEY M" > /tmp/test_inject.txt
sleep 0.5

# Move to center and hold mouse for a very long time (10 seconds)
# With the old system, it would shrink. With continuous source, it should not!
echo "HOLD_MS 10000" > /tmp/test_inject.txt

