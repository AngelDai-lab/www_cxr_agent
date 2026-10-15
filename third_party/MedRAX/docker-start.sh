#!/bin/bash
# MedRAX Docker Quick Start Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🤖 MedRAX Docker Setup${NC}"
echo "================================"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${RED}❌ Docker Compose is not installed. Please install Docker Compose first.${NC}"
    exit 1
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}⚠️  .env file not found. Creating from .env.example...${NC}"
    if [ -f .env.example ]; then
        cp .env.example .env
        echo -e "${YELLOW}📝 Please edit .env and add your OPENAI_API_KEY${NC}"
        echo -e "${YELLOW}   nano .env${NC}"
        read -p "Press Enter after setting up your .env file..."
    else
        echo -e "${RED}❌ .env.example not found!${NC}"
        exit 1
    fi
fi

# Check if OPENAI_API_KEY is set
source .env
if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "your-openai-api-key-here" ]; then
    echo -e "${RED}❌ OPENAI_API_KEY is not set in .env file${NC}"
    echo -e "${YELLOW}   Please edit .env and set your API key${NC}"
    exit 1
fi

# Create necessary directories
echo -e "${GREEN}📁 Creating directories...${NC}"
mkdir -p temp logs
chmod 777 temp logs

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✅ NVIDIA GPU detected${NC}"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
else
    echo -e "${YELLOW}⚠️  No NVIDIA GPU detected. Running in CPU mode may be slow.${NC}"
fi

# Build the Docker image
echo -e "${GREEN}🔨 Building Docker image...${NC}"
docker-compose build

# Start the service
echo -e "${GREEN}🚀 Starting MedRAX service...${NC}"
docker-compose up -d

# Wait for service to be ready
echo -e "${GREEN}⏳ Waiting for service to start...${NC}"
sleep 10

# Check if service is running
if docker-compose ps | grep -q "Up"; then
    echo -e "${GREEN}✅ MedRAX is running!${NC}"
    echo ""
    echo -e "${GREEN}🌐 Access the application at:${NC}"
    echo -e "   ${YELLOW}http://localhost:8585${NC}"
    echo ""
    echo -e "${GREEN}📊 View logs:${NC}"
    echo -e "   ${YELLOW}docker-compose logs -f medrax${NC}"
    echo ""
    echo -e "${GREEN}🛑 Stop the service:${NC}"
    echo -e "   ${YELLOW}docker-compose down${NC}"
else
    echo -e "${RED}❌ Failed to start MedRAX${NC}"
    echo -e "${YELLOW}Check logs with: docker-compose logs medrax${NC}"
    exit 1
fi
