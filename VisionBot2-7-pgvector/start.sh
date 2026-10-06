#!/bin/bash

# Start backend
.venv/bin/uvicorn api.server:app --reload --port 8000 &
BACKEND_PID=$!
echo "Backend started (PID $BACKEND_PID)"

# Start frontend
cd frontend && npm run dev &
FRONTEND_PID=$!
echo "Frontend started (PID $FRONTEND_PID)"

echo ""
echo "Backend:  http://localhost:8000"
echo "Frontend: http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop both"

# Kill both on Ctrl+C
trap "kill $BACKEND_PID $FRONTEND_PID; exit" INT
wait
