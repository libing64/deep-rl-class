#!/bin/bash
# Quick launcher for LunarLander simulation

echo "========================================"
echo "  LunarLander-v2 Simulation Launcher"
echo "========================================"
echo ""
echo "Select mode:"
echo "  1. Real-time display (pygame window)"
echo "  2. Plot analysis (matplotlib charts)"
echo "  3. Both modes"
echo "  4. Exit"
echo ""
read -p "Enter choice (1-4): " choice

case $choice in
    1)
        echo "Starting real-time display mode..."
        echo "1" | conda run -n deep-rl-class python lunar_lander_simulation.py
        ;;
    2)
        echo "Starting plot analysis mode..."
        echo "2" | conda run -n deep-rl-class python lunar_lander_simulation.py
        ;;
    3)
        echo "Starting both modes..."
        echo "3" | conda run -n deep-rl-class python lunar_lander_simulation.py
        ;;
    4)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid choice. Using default (real-time display)..."
        echo "1" | conda run -n deep-rl-class python lunar_lander_simulation.py
        ;;
esac

echo ""
echo "Simulation complete!"

