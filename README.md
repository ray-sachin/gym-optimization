# GymFlow Optimization Project

**Algorithmic Optimization of Gym Usage in Resource-Constrained Campus Environments**

### Team Members
- **Sachin Ray** (Roll No: BT24CSE001)
- **Ankit Patel** (Roll No: BT24CSE005)

### Problem Statement
The university infrastructure hosts two fitness centres with asymmetric accessibility and demographic constraints. The system currently operates on a Greedy / First-Come-First-Served (FCFS) basis, leading to compounding inefficiencies such as **congestion** and over-utilisation of Gym A during evening peak hours, exceeding the equipment-to-user ratio.

This archive contains the core implementation for the Gym Optimization system, focusing on efficient slot allocation to resolve these inefficiencies, maximize student satisfaction, and dynamically balance the load across all available time slots.

## Contents of this Archive

1. **`gym_optimization.py`**
   - The core algorithmic engine of the project.
   - Contains the `Student` data model and scoring systems.
   - Implements the three primary algorithmic stages: 
     - **Stage 1:** Edmonds-Karp Network Flow (for theoretical max-matching)
     - **Stage 2:** Min-Heap Priority Scheduling (for handling initial assignment and waiting lists)
     - **Stage 3:** First-Fit Decreasing (FFD) Bin Packing (for proactive load rebalancing)
   - Includes the **Dynamic Peak Detection** logic, which calculates peak hours dynamically based on live student demand (≥30% threshold).

2. **`run.py`**
   - The main interactive execution script.
   - Accepts custom student inputs and executes the 3-stage optimization pipeline.
   - Features the **Interactive Relocation Consent** system with a Juniors-First policy, allowing juniors to claim slots temporarily held by seniors.

3. **`DAA_Project_GymOptimization.pdf`**
   - The formal project report detailing the problem statement, system architecture, algorithmic complexity (Big-O analysis), and conclusions.

4. **`demo_output.txt`**
   - A perfectly formatted sample terminal output demonstrating the complete flow of the algorithm using a 5-student test case.

5. **`Output Images/`**
   - A directory containing screenshots demonstrating the system output and terminal interaction.

## How to Run

To run the simulation and view the algorithm in action:

1. Ensure you have Python 3 installed on your system.
2. Open a terminal or command prompt in this directory.
3. Execute the main script:
   ```bash
   python3 run.py
   ```
4. Follow the interactive prompts to enter student details.
5. Review the 3-stage allocation process, load imbalance metrics, and test the interactive relocation consent menus.

## Key Features & Scoring

- **Dynamic Peak Detection:** Peak slots are no longer hardcoded (e.g., 5 PM) but are determined by analyzing student input preferences.
- **Juniors-First Relocation:** Displaced students are re-accommodated based on year (Year 1 → Year 4). Juniors can claim slots from displaced seniors, who are then safely re-queued.
- **Satisfaction Formula:** 
  - **Base Score:** 1.0 (Perfect satisfaction)
  - **Travel Time Penalty:** -0.005 per minute of commute
  - **Relocation Penalty:** -0.02 per hour of deviation from the preferred slot
