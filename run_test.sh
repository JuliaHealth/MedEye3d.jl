xvfb-run -s "-screen 0 1024x768x24" julia --project=. scripts/run_interactive_mrb.jl &
PID=$!
sleep 15
kill -9 $PID
