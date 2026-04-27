"""
=============================================================================
 DAA Project: Algorithmic Optimization of Gym Usage in
               Resource-Constrained Campus Environments
 Authors     : Sachin Ray (BT24CSE001) | Ankit Patel (BT24CSE005)
 Subject     : Design and Analysis of Algorithms
 Faculty     : Dr. Sneha Chauhan
=============================================================================

 Three-stage pipeline:
   1. Network Flow (Edmonds-Karp)   — determines max schedulable students
   2. Priority Scheduling (Min-Heap) — assigns each student a gym + slot
   3. Bin Packing / Load Balancing   — rebalances overflow via FFD heuristic

 Objective Function:
   U = Σ(satisfaction) − λ·(peak congestion) − γ·(avg travel time)
=============================================================================
"""

from collections import defaultdict, deque
import math
import heapq
import random

# ─────────────────────────────────────────────────────────────────────────────
# Constants — time slots are represented as integers (hour of the day)
# ─────────────────────────────────────────────────────────────────────────────
MORNING_SLOTS = [6, 7, 8]               # 06:00–09:00
EVENING_SLOTS = [16, 17, 18, 19]        # 16:00–20:00
ALL_SLOTS     = MORNING_SLOTS + EVENING_SLOTS  # 7 one-hour slots total
PEAK_SLOTS    = [17, 18]                # default fallback only


def compute_peak_slots(students, threshold=0.3):
    """
    Dynamically determine which slots are 'peak' based on actual student demand.
    A slot is peak if it attracts >= threshold fraction of all students.
    E.g., if 30% or more students prefer slot 7:00, then 7:00 is peak.
    Falls back to the hardcoded PEAK_SLOTS if no students are provided.
    """
    if not students:
        return list(PEAK_SLOTS)
    from collections import Counter
    pref_counts = Counter(s.preferred_slot for s in students)
    total = len(students)
    dynamic_peaks = [slot for slot, count in pref_counts.items()
                     if count / total >= threshold]
    return sorted(dynamic_peaks) if dynamic_peaks else list(PEAK_SLOTS)

# Penalty coefficients for the objective function (tunable policy levers)
LAMBDA = 0.4   # overcrowding penalty weight
GAMMA  = 0.2   # travel-time penalty weight


# ─────────────────────────────────────────────────────────────────────────────
# Student Data Model
# ─────────────────────────────────────────────────────────────────────────────
class Student:
    """
    Represents a single student requesting a gym slot.

    Attributes:
        name           : unique identifier (e.g. 'S01')
        gender         : 'M' or 'F'  (determines gym eligibility)
        year           : 1 = first-year (lives far from Gym A), else other
        preferred_slot : the hour they *want* to work out (e.g. 17 = 5 PM)
        travel_time    : one-way travel time to assigned gym (minutes)
        assigned_gym   : set after scheduling ('A' or 'B')
        assigned_slot  : set after scheduling (hour integer)
    """
    def __init__(self, name, gender, year, preferred_slot, travel_time):
        self.name           = name
        self.gender         = gender          # 'M' or 'F'
        self.year           = year            # 1 = first-year
        self.preferred_slot = preferred_slot  # desired hour
        self.travel_time    = travel_time     # minutes to gym
        self.assigned_gym   = None
        self.assigned_slot  = None

    def satisfaction_score(self):
        """
        Individual satisfaction ∈ [0, 1].
        Penalises slot deviation (−0.05 per hour) and travel time (−0.02/min).
        """
        if self.assigned_slot is None:
            return 0.0
        slot_deviation = abs(self.assigned_slot - self.preferred_slot)
        score = 1.0 - 0.02 * slot_deviation - 0.005 * self.travel_time
        return max(0.0, round(score, 3))

    def __repr__(self):
        return (f"Student({self.name}, {self.gender}, yr{self.year}, "
                f"pref={self.preferred_slot}, travel={self.travel_time}min, "
                f"assigned={self.assigned_gym}@{self.assigned_slot})")


# ═════════════════════════════════════════════════════════════════════════════
#  ALGORITHM 1 — Network Flow Optimisation (Edmonds-Karp)
# ═════════════════════════════════════════════════════════════════════════════

class FlowNetwork:
    """
    Directed graph storing residual capacities as a nested dict.
    graph[u][v] = remaining capacity from u to v.

    Uses Edmonds-Karp (BFS-based Ford-Fulkerson) to compute max flow.
    Time  Complexity: O(V · E²) where V = vertices, E = edges
    Space Complexity: O(V + E) for the adjacency-list residual graph
    """

    def __init__(self):
        # defaultdict(lambda: defaultdict(int)) gives 0 for missing edges
        self.graph = defaultdict(lambda: defaultdict(int))

    def add_edge(self, u, v, capacity):
        """Add a forward edge u→v with the given capacity.
           A reverse edge v→u (capacity 0) is created for the residual graph."""
        self.graph[u][v] += capacity
        self.graph[v][u] += 0        # ensure reverse edge exists

    def _bfs(self, source, sink, parent):
        """
        BFS finds an augmenting path from source to sink.
        Stores the path in the `parent` dict.
        Returns True if an augmenting path exists.
        Time: O(V + E) per call.
        """
        visited = {source}
        queue   = deque([source])
        while queue:
            node = queue.popleft()
            for neighbour, cap in self.graph[node].items():
                if neighbour not in visited and cap > 0:
                    visited.add(neighbour)
                    parent[neighbour] = node
                    if neighbour == sink:
                        return True          # path found
                    queue.append(neighbour)
        return False                         # no augmenting path

    def max_flow(self, source, sink):
        """
        Edmonds-Karp: repeatedly find shortest augmenting paths via BFS,
        push flow along them, until no more augmenting paths exist.
        Returns the maximum flow value.
        """
        total = 0
        while True:
            parent = {}
            if not self._bfs(source, sink, parent):
                break                        # optimum reached

            # Find bottleneck capacity along the found path
            bottleneck, node = math.inf, sink
            while node != source:
                bottleneck = min(bottleneck, self.graph[parent[node]][node])
                node = parent[node]

            # Push flow: subtract from forward edges, add to backward edges
            node = sink
            while node != source:
                self.graph[parent[node]][node] -= bottleneck
                self.graph[node][parent[node]] += bottleneck
                node = parent[node]

            total += bottleneck

        return total


def build_and_solve_flow_network(students, slot_capacity=8):
    """
    Construct the flow network from students and solve for max flow.

    Network structure:
      SOURCE → Hostel clusters → Gyms → Time-slot nodes → SINK

    Gender exclusivity is enforced by edge design:
      - Female hostel (H_F) connects ONLY to GYM_B
      - Male hostels connect ONLY to GYM_A

    Returns (max_flow_value, FlowNetwork_object).
    """
    fn = FlowNetwork()

    # Classify students into demographic clusters
    male_1yr   = [s for s in students if s.gender == 'M' and s.year == 1]
    male_other = [s for s in students if s.gender == 'M' and s.year != 1]
    female     = [s for s in students if s.gender == 'F']

    # SOURCE → hostel cluster nodes (capacity = cluster size)
    fn.add_edge('SOURCE', 'H_1yr_M',   len(male_1yr))
    fn.add_edge('SOURCE', 'H_other_M', len(male_other))
    fn.add_edge('SOURCE', 'H_F',       len(female))

    # Hostel → gym nodes (gender exclusivity built into edges)
    fn.add_edge('H_1yr_M',   'GYM_A', len(male_1yr))
    fn.add_edge('H_other_M', 'GYM_A', len(male_other))
    fn.add_edge('H_F',       'GYM_B', len(female))      # females → Gym B only

    # Gyms → time-slot nodes → SINK (capacity = slot_capacity per slot)
    for slot in ALL_SLOTS:
        fn.add_edge('GYM_A', f'GYM_A_s{slot}', slot_capacity)
        fn.add_edge('GYM_B', f'GYM_B_s{slot}', slot_capacity)
        fn.add_edge(f'GYM_A_s{slot}', 'SINK', slot_capacity)
        fn.add_edge(f'GYM_B_s{slot}', 'SINK', slot_capacity)

    return fn.max_flow('SOURCE', 'SINK'), fn


# ═════════════════════════════════════════════════════════════════════════════
#  ALGORITHM 2 — Priority-Based Scheduling (Min-Heap)
# ═════════════════════════════════════════════════════════════════════════════

# Priority values (lower = higher priority in a min-heap)
PRIORITY = {
    '1yr_offpeak' : 1,   # Highest — 1st-year choosing off-peak (incentive)
    '1yr_peak'    : 2,   # 1st-year wanting peak (high travel cost)
    'other_offpeak': 3,  # Other students choosing off-peak
    'other_peak'  : 4,   # Lowest — other students wanting peak
}


def get_priority(student, slot, peak_slots=None):
    """
    Compute scheduling priority based on student type and slot choice.
    1st-year students who voluntarily choose off-peak get highest priority,
    creating an incentive structure that naturally redistributes load.
    peak_slots is dynamically computed from student demand.
    """
    if peak_slots is None:
        peak_slots = PEAK_SLOTS
    is_1yr    = student.year == 1
    is_offpeak = slot not in peak_slots
    if is_1yr and is_offpeak:   return PRIORITY['1yr_offpeak']
    elif is_1yr:                return PRIORITY['1yr_peak']
    elif is_offpeak:            return PRIORITY['other_offpeak']
    else:                       return PRIORITY['other_peak']


class GymSlotScheduler:
    """
    Min-Heap priority scheduler with slot relaxation.

    Algorithm Used : Min-Heap Priority Queue (Python heapq)
    Time Complexity: O(n log n) — each of n students inserted/extracted once
    Space Complexity: O(n) for the heap

    Slot Relaxation: if a student's preferred slot is full, the scheduler
    searches outward (±1, ±2, ...) for the nearest available alternative,
    minimising dissatisfaction.
    """

    def __init__(self, gym_capacities, slot_capacity, peak_slots=None):
        self.slot_capacity = slot_capacity
        self.peak_slots = peak_slots if peak_slots is not None else list(PEAK_SLOTS)
        # slot_load[gym_id][slot_hour] tracks current occupancy
        self.slot_load = {
            gid: {s: 0 for s in ALL_SLOTS}
            for gid in gym_capacities
        }
        self._heap = []
        self._seq  = 0            # tie-breaker for equal priorities

    def submit_request(self, student):
        """Insert student into the min-heap with computed priority. O(log n)."""
        p = get_priority(student, student.preferred_slot, self.peak_slots)
        self._seq += 1
        heapq.heappush(self._heap, (p, self._seq, student.preferred_slot, student))

    def _find_slot(self, student, preferred):
        """
        Slot relaxation: try preferred slot first, then nearest alternatives.
        Search is performed across all allowed slots using actual time-distance,
        so 8:00 can still fall through to 16:00 when the morning block is full,
        while 16:00 still prefers 17:00 over jumping straight to 8:00.
        Returns (gym_id, slot) or (None, None) if all slots are full.
        """
        # Gender determines eligible gyms
        eligible = ['A'] if student.gender == 'M' else ['B']

        # Search all allowed slots by increasing time-distance from preferred.
        # Lower clock times win ties to keep the ordering deterministic.
        search = sorted(
            ALL_SLOTS,
            key=lambda slot: (abs(slot - preferred), slot),
        )

        # Try each slot in search order for each eligible gym
        for slot in search:
            for gym in eligible:
                if self.slot_load[gym][slot] < self.slot_capacity:
                    return gym, slot
        return None, None

    def process_all(self):
        """
        Drain the heap and assign students to slots. O(n log n) total.
        Returns (assigned_list, waiting_list).
        """
        assigned, waiting = [], []
        while self._heap:
            _, _, preferred, student = heapq.heappop(self._heap)
            gym, slot = self._find_slot(student, preferred)
            if gym:
                self.slot_load[gym][slot] += 1
                student.assigned_gym, student.assigned_slot = gym, slot
                assigned.append(student)
            else:
                waiting.append(student)
        return assigned, waiting


# ═════════════════════════════════════════════════════════════════════════════
#  ALGORITHM 3 — Bin Packing / Load Balancing (FFD Heuristic)
# ═════════════════════════════════════════════════════════════════════════════

class BinPackingBalancer:
    """
    First Fit Decreasing (FFD) inspired load rebalancer.

    Bins  = time slots (each with capacity C_j)
    Items = students (weight 1 each)

    Migrates overflow students from peak slots to the lightest morning slot.
    Approximation guarantee: FFD uses at most 11/9 · OPT + 6/9 bins.

    Time Complexity : O(n · S) where S = 7 slots ≈ O(n)
    Space Complexity: O(n + S)
    """

    def __init__(self, slot_capacity, peak_slots=None):
        self.slot_capacity = slot_capacity
        self.peak_slots = peak_slots if peak_slots is not None else list(PEAK_SLOTS)

    def rebalance(self, assigned_students, slot_load):
        """
        Proactive Load Spreading (First Fit Decreasing inspired).

        Instead of only fixing overflows, this actively balances load
        across all slots within each gym by migrating already-relocated
        students from congested (above-average) slots to lighter
        (below-average) slots.

        Only students who are NOT at their preferred slot are eligible
        for migration (we never move a student who got their first choice).

        Returns (migrated_count, updated_slot_load).
        """
        migrated = 0

        for gid in slot_load:
            # Calculate average load for this gym
            loads = slot_load[gid]
            total_load = sum(loads.values())
            if total_load == 0:
                continue
            avg_load = total_load / len(ALL_SLOTS)

            # Identify heavy slots (above average) and light slots (below average)
            heavy_slots = sorted(
                [s for s in ALL_SLOTS if loads[s] > avg_load],
                key=lambda s: loads[s],
                reverse=True   # heaviest first (FFD order)
            )
            light_slots = sorted(
                [s for s in ALL_SLOTS if loads[s] < avg_load and
                 loads[s] < self.slot_capacity],
                key=lambda s: loads[s]   # lightest first
            )

            if not heavy_slots or not light_slots:
                continue

            # Collect movable candidates from heavy slots:
            # only students already relocated (assigned != preferred)
            candidates = []
            for student in assigned_students:
                if (student.assigned_gym == gid and
                    student.assigned_slot in heavy_slots and
                    student.assigned_slot != student.preferred_slot):
                    # Score by distance from preferred — move the MOST
                    # displaced students first (they lose the least)
                    dist = abs(student.assigned_slot - student.preferred_slot)
                    candidates.append((dist, student.name, student))

            # Sort: most displaced first (FFD: largest "misfit" items first)
            candidates.sort(key=lambda x: x[0], reverse=True)

            for _, _, student in candidates:
                # Recheck: is the source slot still heavy?
                if loads[student.assigned_slot] <= avg_load:
                    continue

                # Find the best light slot (closest to student's preference)
                best_slot = None
                best_dist = 999
                for ls in light_slots:
                    if loads[ls] < self.slot_capacity and loads[ls] < avg_load:
                        d = abs(ls - student.preferred_slot)
                        if d < best_dist:
                            best_dist = d
                            best_slot = ls

                if best_slot is not None:
                    old_slot = student.assigned_slot
                    loads[old_slot] -= 1
                    loads[best_slot] += 1
                    student.assigned_slot = best_slot
                    migrated += 1

        return migrated, slot_load

    @staticmethod
    def load_imbalance_index(slot_load):
        """
        LII = (max_load − min_load) / avg_load across all gym-slot pairs.
        Lower is better (perfectly balanced → 0).
        """
        loads = [v for gym in slot_load.values() for v in gym.values()]
        if not loads or sum(loads) == 0:
            return 0.0
        avg = sum(loads) / len(loads)
        return round((max(loads) - min(loads)) / avg, 3) if avg else 0.0


# ═════════════════════════════════════════════════════════════════════════════
#  OBJECTIVE FUNCTION
# ═════════════════════════════════════════════════════════════════════════════

def compute_objective(assigned, slot_load, slot_capacity, peak_slots=None):
    if peak_slots is None:
        peak_slots = PEAK_SLOTS
    """
    Global utility:  U = Σ(satisfaction) − λ·(peak congestion) − γ·(avg travel)

    Where:
      Σ(satisfaction)  = sum of individual satisfaction scores ∈ [0, 1]
      Peak Congestion  = Σ max(0, load_t − C_j)  for all peak slots t
      Avg Travel Time  = mean travel time (minutes) across assigned students

    Returns a dict with 'U', 'satisfaction', 'congestion', 'travel'.
    """
    total_satisfaction = sum(s.satisfaction_score() for s in assigned)
    peak_congestion = sum(
        max(0, load - slot_capacity)
        for gym_loads in slot_load.values()
        for slot, load in gym_loads.items()
        if slot in peak_slots
    )
    avg_travel = (
        sum(s.travel_time for s in assigned) / len(assigned)
        if assigned else 0
    )
    U = total_satisfaction - LAMBDA * peak_congestion - GAMMA * avg_travel
    return {
        'U':            round(U, 3),
        'satisfaction': round(total_satisfaction, 3),
        'congestion':   peak_congestion,
        'travel':       round(avg_travel, 3)
    }


# ═════════════════════════════════════════════════════════════════════════════
#  STUDENT GENERATOR (for testing / simulation)
# ═════════════════════════════════════════════════════════════════════════════

def generate_students(n_1yr_male=20, n_other_male=40, n_female=25, seed=42):
    """
    Generate a realistic student population.
    - 1st-year males  : travel_time=10 min (far hostel), 70% prefer peak
    - Other males     : travel_time=2 min  (nearby),     70% prefer peak
    - Females         : travel_time=2 min  (nearby),     70% prefer peak
    """
    random.seed(seed)
    students = []
    sid = 1

    def make(n, gender, year, travel):
        nonlocal sid
        for _ in range(n):
            # 70% bias toward peak slots (17:00–18:00), 30% off-peak
            if random.random() < 0.7:
                pref = random.choice([17, 18])
            else:
                pref = random.choice([6, 7, 8, 16, 19])
            students.append(Student(f'S{sid:03d}', gender, year, pref, travel))
            sid += 1

    make(n_1yr_male,  'M', 1, 10)   # 1st-year males, far hostel
    make(n_other_male,'M', 2,  2)   # other males, nearby
    make(n_female,    'F', 2,  2)   # females, nearby
    return students


# ═════════════════════════════════════════════════════════════════════════════
#  FCFS BASELINE (for comparison)
# ═════════════════════════════════════════════════════════════════════════════

def fcfs_baseline(students, slot_capacity=8):
    """
    Naive First-Come-First-Served scheduler (no priority, no relaxation).
    Students are processed in random arrival order; each gets their
    preferred slot if capacity allows, otherwise they are denied.
    """
    import copy
    pool = copy.deepcopy(students)
    random.shuffle(pool)

    slot_load = {'A': {s: 0 for s in ALL_SLOTS},
                 'B': {s: 0 for s in ALL_SLOTS}}
    assigned, denied = [], []

    for st in pool:
        gym = 'A' if st.gender == 'M' else 'B'
        slot = st.preferred_slot
        if slot_load[gym][slot] < slot_capacity:
            slot_load[gym][slot] += 1
            st.assigned_gym, st.assigned_slot = gym, slot
            assigned.append(st)
        else:
            denied.append(st)

    return assigned, denied, slot_load


# ═════════════════════════════════════════════════════════════════════════════
#  FULL PIPELINE — runs all three algorithms in sequence
# ═════════════════════════════════════════════════════════════════════════════

# ═════════════════════════════════════════════════════════════════════════════
#  RELOCATION MANAGER — handles student relocations with cascading allocation
# ═════════════════════════════════════════════════════════════════════════════

class RelocationManager:
    """
    Manages student gym relocations with:
    1. Confirmation prompt for relocation willingness
    2. Slot selection (manual or auto)
    3. Cascading allocation when slot is cancelled
    
    When a student is relocated (cancels current slot):
    - Their slot goes to someone with that slot as preference in waiting list
    - If no preference match, offer to other waiting students
    - Auto-assign uses preference-matching with load balancing
    """
    
    def __init__(self, slot_capacity):
        self.slot_capacity = slot_capacity
    
    def get_empty_slots(self, assigned_students, slot_load):
        """
        Get all empty slots grouped by gym.
        Returns dict: {'A': [slots], 'B': [slots]}
        """
        empty_slots = {'A': [], 'B': []}
        for gym in ['A', 'B']:
            for slot in ALL_SLOTS:
                if slot_load[gym][slot] < self.slot_capacity:
                    empty_slots[gym].append(slot)
        return empty_slots
    
    def get_available_slots_for_student(self, student, assigned_students, slot_load):
        """
        Get available slots for a specific student (respecting gender constraint).
        Returns list of available slots. Empty list if no slots available.
        """
        gym = 'A' if student.gender == 'M' else 'B'
        available = []
        for slot in ALL_SLOTS:
            if slot_load[gym][slot] < self.slot_capacity:
                available.append(slot)
        return sorted(available)
    
    def auto_allocate(self, student, assigned_students, waiting_students, slot_load):
        """
        Automatically allocate best available slot for student.
        Strategy: 
        1. Try preferred slot if available
        2. Otherwise use relaxation (nearest available)
        3. Prefer less-loaded slots for load balancing
        
        Returns (gym, slot) or (None, None) if no slots available.
        """
        gym = 'A' if student.gender == 'M' else 'B'
        
        # Strategy: search by distance from preference, breaking ties by load
        search = sorted(
            ALL_SLOTS,
            key=lambda slot: (abs(slot - student.preferred_slot), slot_load[gym][slot])
        )
        
        for slot in search:
            if slot_load[gym][slot] < self.slot_capacity:
                return gym, slot
        return None, None
    
    def find_preference_match(self, gym, slot, waiting_students):
        """
        Find first waiting student who has this gym/slot as preference.
        Returns (student, index) or (None, None) if no match found.
        """
        for idx, student in enumerate(waiting_students):
            student_gym = 'A' if student.gender == 'M' else 'B'
            if student_gym == gym and student.preferred_slot == slot:
                return student, idx
        return None, None
    
    def process_relocation_yes(self, student, target_gym, target_slot, assigned_students, waiting_students, slot_load):
        """
        Process when student accepts relocation to new slot.
        Removes from current slot, adds to target slot.
        Returns updated lists and slot_load.
        """
        # Remove from old slot
        if student.assigned_slot is not None:
            slot_load[student.assigned_gym][student.assigned_slot] -= 1
        
        # Assign to new slot
        student.assigned_gym = target_gym
        student.assigned_slot = target_slot
        slot_load[target_gym][target_slot] += 1
        
        return assigned_students, waiting_students, slot_load
    
    def process_slot_cancellation(self, student, assigned_students, waiting_students, slot_load):
        """
        Process when student declines relocation (cancels their slot).
        Cascade: Try to give slot to:
        1. First: Waiting student with this slot as preference
        2. Second: First waiting student (offer them the slot)
        
        Returns (assigned, waiting, slot_load, reassignment_info)
        reassignment_info contains details of who got the slot
        """
        cancelled_gym = student.assigned_gym
        cancelled_slot = student.assigned_slot
        reassignment_info = {}
        
        # Remove student from assigned
        assigned_students = [s for s in assigned_students if s.name != student.name]
        
        # Update slot load
        slot_load[cancelled_gym][cancelled_slot] -= 1
        
        # Try to find preference match in waiting list
        preference_match, match_idx = self.find_preference_match(cancelled_gym, cancelled_slot, waiting_students)
        
        if preference_match:
            # Assign slot to preference-matched student
            preference_match.assigned_gym = cancelled_gym
            preference_match.assigned_slot = cancelled_slot
            slot_load[cancelled_gym][cancelled_slot] += 1
            assigned_students.append(preference_match)
            waiting_students.pop(match_idx)
            reassignment_info = {
                'type': 'preference_match',
                'recipient': preference_match.name,
                'gym': cancelled_gym,
                'slot': cancelled_slot,
            }
        elif waiting_students:
            # Offer to first waiting student
            first_waiting = waiting_students[0]
            first_waiting.assigned_gym = cancelled_gym
            first_waiting.assigned_slot = cancelled_slot
            slot_load[cancelled_gym][cancelled_slot] += 1
            assigned_students.append(first_waiting)
            waiting_students.pop(0)
            reassignment_info = {
                'type': 'waiting_list_offer',
                'recipient': first_waiting.name,
                'gym': cancelled_gym,
                'slot': cancelled_slot,
            }
        else:
            # No one to give the slot to
            reassignment_info = {
                'type': 'slot_freed',
                'gym': cancelled_gym,
                'slot': cancelled_slot,
            }
        
        return assigned_students, waiting_students, slot_load, reassignment_info


def run_pipeline(students, slot_capacity=8, verbose=True):
    """
    Execute the complete three-stage optimisation pipeline:
      Stage 1: Edmonds-Karp network flow  → theoretical max schedulable
      Stage 2: Min-Heap priority scheduler → actual slot assignments
      Stage 3: FFD bin-packing rebalancer  → flatten peak overflows

    Returns a results dict with all metrics.
    """
    sep = "=" * 70

    # ── Stage 1: Network Flow ──
    max_flow_val, fn = build_and_solve_flow_network(students, slot_capacity)
    theoretical_max = len(ALL_SLOTS) * slot_capacity * 2  # 2 gyms

    if verbose:
        print(f"\n{sep}")
        print("  STAGE 1 — Network Flow (Edmonds-Karp)")
        print(sep)
        print(f"  Total students        : {len(students)}")
        print(f"  Max flow (schedulable): {max_flow_val}")
        print(f"  Theoretical maximum   : {theoretical_max}")
        print(f"  Flow utilisation      : {round(max_flow_val/theoretical_max*100,1)}%")

    # ── Stage 2: Priority Scheduling ──
    scheduler = GymSlotScheduler({'A': None, 'B': None}, slot_capacity)
    for s in students:
        scheduler.submit_request(s)
    assigned, waiting = scheduler.process_all()
    lii_before = BinPackingBalancer.load_imbalance_index(scheduler.slot_load)

    if verbose:
        print(f"\n{sep}")
        print("  STAGE 2 — Priority Scheduling (Min-Heap)")
        print(sep)
        print(f"  Assigned : {len(assigned)}")
        print(f"  Waiting  : {len(waiting)}")
        print(f"  LII (before rebalance): {lii_before}")
        print(f"\n  Slot loads BEFORE rebalancing:")
        for gym in ['A', 'B']:
            loads_str = ", ".join(
                f"{s}:00={scheduler.slot_load[gym][s]}" for s in ALL_SLOTS
            )
            print(f"    Gym {gym}: {loads_str}")

    # ── Stage 3: Bin Packing Rebalance ──
    balancer = BinPackingBalancer(slot_capacity)
    migrated, slot_load = balancer.rebalance(assigned, scheduler.slot_load)
    lii_after = BinPackingBalancer.load_imbalance_index(slot_load)

    if verbose:
        print(f"\n{sep}")
        print("  STAGE 3 — Bin Packing / Load Balancing (FFD)")
        print(sep)
        print(f"  Students migrated     : {migrated}")
        print(f"  LII (after rebalance) : {lii_after}")
        print(f"\n  Slot loads AFTER rebalancing:")
        for gym in ['A', 'B']:
            loads_str = ", ".join(
                f"{s}:00={slot_load[gym][s]}" for s in ALL_SLOTS
            )
            print(f"    Gym {gym}: {loads_str}")

    # ── Objective Function ──
    obj = compute_objective(assigned, slot_load, slot_capacity)

    if verbose:
        print(f"\n{sep}")
        print("  OBJECTIVE FUNCTION RESULTS")
        print(sep)
        print(f"  Global Utility (U)    : {obj['U']}")
        print(f"  Total Satisfaction    : {obj['satisfaction']}")
        print(f"  Peak Congestion       : {obj['congestion']}")
        print(f"  Avg Travel Time (min) : {obj['travel']}")
        print(sep)

    return {
        'max_flow': max_flow_val,
        'assigned': assigned,
        'waiting': waiting,
        'migrated': migrated,
        'slot_load': slot_load,
        'objective': obj,
        'lii_before': lii_before,
        'lii_after': lii_after,
    }
