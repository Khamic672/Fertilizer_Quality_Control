#!/bin/bash
# Fertilizer QC App - Complete Setup Script

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "════════════════════════════════════════════════════════"
echo "   🌾 Fertilizer Quality Control App Setup"
echo "════════════════════════════════════════════════════════"
echo -e "${NC}"

# Check Python
echo -e "${YELLOW}Checking Python...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python 3 not found${NC}"
    exit 1
fi
PYTHON_VERSION=$(python3 --version)
echo -e "${GREEN}✓ $PYTHON_VERSION${NC}"

# Check Node.js
echo -e "${YELLOW}Checking Node.js...${NC}"
if ! command -v node &> /dev/null; then
    echo -e "${RED}✗ Node.js not found${NC}"
    exit 1
fi
NODE_VERSION=$(node --version)
echo -e "${GREEN}✓ Node.js $NODE_VERSION${NC}"

# Create directory structure
echo -e "\n${YELLOW}Creating project structure...${NC}"
mkdir -p backend/models
mkdir -p backend/src/soil_segment
mkdir -p backend/utils
mkdir -p frontend/src/components
mkdir -p frontend/src/assets
mkdir -p frontend/public/examples

touch backend/src/__init__.py
touch backend/src/soil_segment/__init__.py
touch backend/utils/__init__.py

echo -e "${GREEN}✓ Directory structure created${NC}"

# Setup Backend
echo -e "\n${BLUE}════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}   Setting up Backend${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"

cd backend

# Check for model checkpoints
echo -e "\n${YELLOW}Checking for model checkpoints...${NC}"
MISSING_MODELS=0

if [ ! -f "models/unet_best.pth" ]; then
    echo -e "${RED}✗ models/unet_best.pth not found${NC}"
    MISSING_MODELS=1
else
    echo -e "${GREEN}✓ UNet checkpoint found${NC}"
fi

if [ ! -f "models/regression_model.pkl" ]; then
    echo -e "${RED}✗ models/regression_model.pkl not found${NC}"
    MISSING_MODELS=1
else
    echo -e "${GREEN}✓ Regression model found${NC}"
fi

if [ $MISSING_MODELS -eq 1 ]; then
    echo -e "\n${YELLOW}⚠️  Please place your model checkpoints in backend/models/${NC}"
    echo -e "Required files:"
    echo -e "  - backend/models/unet_best.pth"
    echo -e "  - backend/models/regression_model.pkl"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Install Python dependencies
echo -e "\n${YELLOW}Installing Python dependencies...${NC}"
pip install -q Flask flask-cors torch torchvision opencv-python Pillow numpy scikit-learn joblib

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Python dependencies installed${NC}"
else
    echo -e "${RED}✗ Failed to install Python dependencies${NC}"
    exit 1
fi

cd ..

# Setup Frontend
echo -e "\n${BLUE}════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}   Setting up Frontend${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"

cd frontend

echo -e "\n${YELLOW}Installing Node.js dependencies...${NC}"
npm install --silent

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Node.js dependencies installed${NC}"
else
    echo -e "${RED}✗ Failed to install Node.js dependencies${NC}"
    exit 1
fi

cd ..

# Create run script
echo -e "\n${YELLOW}Creating run scripts...${NC}"

# Backend run script
cat > run-backend.sh << 'EOF'
#!/bin/bash
cd backend
echo "🔧 Starting Flask Backend on http://localhost:5000"
python app.py
EOF

# Frontend run script
cat > run-frontend.sh << 'EOF'
#!/bin/bash
cd frontend
echo "🎨 Starting Vue Frontend on http://localhost:5173"
npm run dev
EOF

chmod +x run-backend.sh
chmod +x run-frontend.sh

echo -e "${GREEN}✓ Run scripts created${NC}"

# Summary
echo -e "\n${GREEN}════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}   ✓ Setup Complete!${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════${NC}"

echo -e "\n${YELLOW}To run the application:${NC}"
echo -e "\n1. Start Backend (Terminal 1):"
echo -e "   ${BLUE}./run-backend.sh${NC}"
echo -e "   Backend: http://localhost:5000\n"

echo -e "2. Start Frontend (Terminal 2):"
echo -e "   ${BLUE}./run-frontend.sh${NC}"
echo -e "   Frontend: http://localhost:5173\n"

echo -e "3. Open browser to:"
echo -e "   ${GREEN}http://localhost:5173${NC}\n"

if [ $MISSING_MODELS -eq 1 ]; then
    echo -e "${YELLOW}⚠️  Remember to add your model checkpoints before running!${NC}"
fi

echo -e "\n${BLUE}Happy analyzing! 🔬${NC}\n"