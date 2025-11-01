#!/bin/bash

# Medical Data Processor - Local Development Startup Script
# This script starts both backend and frontend servers

echo "🚀 Starting Medical Data Processor locally..."
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/frontend"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is not installed${NC}"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo -e "${RED}❌ Node.js is not installed${NC}"
    echo "Please install Node.js 14 or higher"
    exit 1
fi

# Check if backend dependencies are installed
if [ ! -d "$BACKEND_DIR/venv" ] && [ ! -f "$BACKEND_DIR/.deps_installed" ]; then
    echo -e "${YELLOW}⚠️  Backend dependencies may not be installed${NC}"
    echo "Run: cd backend && pip install -r requirements.txt"
    echo ""
fi

# Check if frontend dependencies are installed
if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
    echo -e "${YELLOW}⚠️  Frontend dependencies not installed${NC}"
    echo "Installing frontend dependencies..."
    cd "$FRONTEND_DIR"
    npm install
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Failed to install frontend dependencies${NC}"
        exit 1
    fi
    cd "$SCRIPT_DIR"
    echo ""
fi

# Function to cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}🛑 Shutting down servers...${NC}"
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 0
}

# Set trap to cleanup on Ctrl+C
trap cleanup SIGINT SIGTERM

# Start backend
echo -e "${BLUE}📡 Starting backend server on port 8000...${NC}"
cd "$BACKEND_DIR"

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo -e "${GREEN}✓ Using virtual environment${NC}"
    source venv/bin/activate
    python main.py > /tmp/medical-backend.log 2>&1 &
    BACKEND_PID=$!
else
    python3 main.py > /tmp/medical-backend.log 2>&1 &
    BACKEND_PID=$!
fi

# Wait for backend to start
sleep 3

# Check if backend is running
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo -e "${RED}❌ Backend failed to start${NC}"
    echo "Check logs: tail -f /tmp/medical-backend.log"
    exit 1
fi

# Test backend health
if curl -s http://localhost:8000 > /dev/null; then
    echo -e "${GREEN}✓ Backend started successfully at http://localhost:8000${NC}"
else
    echo -e "${YELLOW}⚠️  Backend started but health check failed${NC}"
    echo "It may still be starting up..."
fi

echo ""

# Start frontend
echo -e "${BLUE}🎨 Starting frontend server on port 8080...${NC}"
cd "$FRONTEND_DIR"
npm run serve > /tmp/medical-frontend.log 2>&1 &
FRONTEND_PID=$!

# Wait for frontend to start
sleep 5

# Check if frontend is running
if ! kill -0 $FRONTEND_PID 2>/dev/null; then
    echo -e "${RED}❌ Frontend failed to start${NC}"
    echo "Check logs: tail -f /tmp/medical-frontend.log"
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

echo -e "${GREEN}✓ Frontend started successfully${NC}"
echo ""

# Display status
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ Medical Data Processor is running!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  ${BLUE}Frontend:${NC}  http://localhost:8080"
echo -e "  ${BLUE}Backend:${NC}   http://localhost:8000"
echo -e "  ${BLUE}API Docs:${NC}  http://localhost:8000/docs"
echo ""
echo -e "${YELLOW}📋 Logs:${NC}"
echo -e "  Backend:  tail -f /tmp/medical-backend.log"
echo -e "  Frontend: tail -f /tmp/medical-frontend.log"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop all servers${NC}"
echo ""

# Keep script running and show backend logs
tail -f /tmp/medical-backend.log

