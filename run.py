"""
=============================================================================
 Interactive Gym Scheduler with Relocation Consent
 
 After the pipeline assigns students, any student who got RELOCATED
 (assigned slot ≠ preferred slot) is prompted interactively:
   • YES  → pick from available slots manually, or choose AUTO
   • NO   → slot is cancelled; cascading logic gives it to a waiting
             student who prefers that slot, or the next in the queue
             
 Usage:
     python3 run.py
=============================================================================
"""

from gym_optimization import (
    Student, GymSlotScheduler, BinPackingBalancer,
    build_and_solve_flow_network, compute_objective,
    RelocationManager, ALL_SLOTS, PEAK_SLOTS, compute_peak_slots
)

VALID_SLOTS = [6, 7, 8, 16, 17, 18, 19]
MAX_YEAR = 4
SEP = "=" * 70
THIN = "-" * 70


# ─────────────────────────────────────────────────────────────────────────────
#  Input helpers
# ─────────────────────────────────────────────────────────────────────────────

def read_int(prompt, minimum=None, maximum=None):
    """Read an integer with optional bounds validation."""
    while True:
        try:
            value = int(input(prompt).strip())
            if minimum is not None and value < minimum:
                print(f"  ⚠ Value must be at least {minimum}.")
                continue
            if maximum is not None and value > maximum:
                print(f"  ⚠ Value must be at most {maximum}.")
                continue
            return value
        except ValueError:
            print("  ⚠ Please enter a valid integer.")


def read_gender(prompt):
    """Read M or F."""
    while True:
        value = input(prompt).strip().upper()
        if value in {"M", "F"}:
            return value
        print("  ⚠ Please enter M for Male or F for Female.")


def read_slot(prompt):
    """Read a valid slot hour."""
    while True:
        value = read_int(prompt)
        if value in VALID_SLOTS:
            return value
        print(f"  ⚠ Must be one of: {', '.join(map(str, VALID_SLOTS))}")


def read_yes_no(prompt):
    """Read y/yes or n/no (case-insensitive)."""
    while True:
        value = input(prompt).strip().lower()
        if value in {"y", "yes"}:
            return True
        if value in {"n", "no"}:
            return False
        print("  ⚠ Please enter y/yes or n/no.")


# ─────────────────────────────────────────────────────────────────────────────
#  Display helpers
# ─────────────────────────────────────────────────────────────────────────────

def print_student_line(s):
    """Print a single student's assignment details."""
    status = "✓ Preferred" if s.assigned_slot == s.preferred_slot else "↻ Relocated"
    print(
        f"  {s.name} | {s.gender} | Year {s.year} | "
        f"Pref={s.preferred_slot}:00 → Gym {s.assigned_gym} @ "
        f"{s.assigned_slot}:00 | Score={s.satisfaction_score()} | {status}"
    )


def print_slot_loads(slot_load):
    """Print current slot load table."""
    print(f"\n  {'Slot':<8}", end="")
    for s in VALID_SLOTS:
        print(f"{s}:00  ", end="")
    print()
    for gym in ["A", "B"]:
        print(f"  Gym {gym}   ", end="")
        for s in VALID_SLOTS:
            print(f" {slot_load[gym][s]:>2}   ", end="")
        print()


def print_full_results(assigned, waiting, slot_load, slot_capacity, peak_slots=None):
    """Print complete results summary."""
    obj = compute_objective(assigned, slot_load, slot_capacity, peak_slots=peak_slots)

    print(f"\n{SEP}")
    print("  FINAL RESULTS")
    print(SEP)
    print(f"  Assigned : {len(assigned)}")
    print(f"  Waiting  : {len(waiting)}")
    print(f"  Utility  : {obj['U']}")
    print(f"  Satisfaction : {obj['satisfaction']}")
    print(f"  Congestion   : {obj['congestion']}")
    print(f"  Avg Travel   : {obj['travel']} min")

    print(f"\n  ASSIGNED STUDENTS")
    print(f"  {THIN}")
    if assigned:
        for s in assigned:
            print_student_line(s)
    else:
        print("  (none)")

    print(f"\n  WAITING LIST")
    print(f"  {THIN}")
    if waiting:
        for s in waiting:
            print(
                f"  {s.name} | {s.gender} | Year {s.year} | "
                f"Pref={s.preferred_slot}:00"
            )
    else:
        print("  (none)")

    print_slot_loads(slot_load)
    print()


# ─────────────────────────────────────────────────────────────────────────────
#  Interactive relocation flow
# ─────────────────────────────────────────────────────────────────────────────

def handle_relocations(assigned, waiting, slot_load, slot_capacity):
    """
    Relocation consent — JUNIORS FIRST policy:
      1. Sort relocated students juniors-first (Year 1 -> Year 4).
      2. Juniors see truly empty slots PLUS slots temporarily held by
         other relocated students (seniors). They can CLAIM those slots.
      3. If a junior claims a senior's slot, the senior is displaced
         and re-queued to pick again after.
      4. Seniors finally pick from whatever slots remain.
    """
    relocator = RelocationManager(slot_capacity)

    relocated = sorted(
        [s for s in assigned if s.assigned_slot != s.preferred_slot],
        key=lambda s: s.year,
        reverse=False   # Year 1 first, Year 4 last
    )

    if not relocated:
        print("\n  ✓ No students were relocated — everyone got their preferred slot!")
        return assigned, waiting, slot_load

    print(f"\n{SEP}")
    print(f"  RELOCATION CONSENT — {len(relocated)} student(s) were relocated")
    print(SEP)
    print("  Processing order: Juniors first → Seniors last")
    print("  Juniors may claim slots temporarily held by relocated seniors.")
    print("  Each student can pick a slot, AUTO-pick, or CANCEL entirely.\n")

    process_queue = list(relocated)
    already_processed = set()

    while process_queue:
        student = process_queue.pop(0)

        if student.name in already_processed:
            continue
        if student not in assigned:
            continue

        already_processed.add(student.name)

        gym     = student.assigned_gym
        current = student.assigned_slot

        print(f"\n  ┌─ {student.name} (Gender={student.gender}, Year={student.year})")
        print(f"  │  Preferred : {student.preferred_slot}:00")
        print(f"  │  Assigned  : Gym {gym} @ {current}:00  (relocated)")
        print(f"  │  Travel    : {student.travel_time} min")
        print(f"  └─ Satisfaction: {student.satisfaction_score()}")

        # Temporarily free own slot so it shows as available
        slot_load[gym][current] -= 1

        # Map (gym, slot) -> holder for students still pending in queue
        pending = {
            (s.assigned_gym, s.assigned_slot): s
            for s in process_queue
            if s.name not in already_processed
            and s.assigned_gym and s.assigned_slot is not None
        }

        truly_empty = []
        senior_held = []
        for slot in ALL_SLOTS:
            if slot_load[gym][slot] < slot_capacity:
                truly_empty.append((slot, None))
            else:
                holder = pending.get((gym, slot))
                if holder:
                    senior_held.append((slot, holder))

        available = truly_empty + senior_held

        print(f"\n  All selectable slots for Gym {gym}:")
        for i, (slot, holder) in enumerate(available, 1):
            load = slot_load[gym][slot]
            tags = []
            if slot == current:
                tags.append("← currently assigned")
            if slot == student.preferred_slot:
                tags.append("★ your preference")
            if holder:
                tags.append(f"held by {holder.name} Yr{holder.year} — claimable")
            tag_str = f"  ({', '.join(tags)})" if tags else ""
            print(f"    [{i}] {slot}:00  (load: {load}/{slot_capacity}){tag_str}")

        print("    [A] AUTO   — system picks the best empty slot for you")
        print("    [C] CANCEL — decline entirely (free your slot)")

        while True:
            choice = input("\n  Your choice: ").strip().upper()

            if choice == "A":
                best_gym, best_slot = relocator.auto_allocate(
                    student, assigned, waiting, slot_load
                )
                if best_gym:
                    student.assigned_gym  = best_gym
                    student.assigned_slot = best_slot
                    slot_load[best_gym][best_slot] += 1
                    print(f"  ✓ Auto-assigned to Gym {best_gym} @ {best_slot}:00")
                else:
                    slot_load[gym][current] += 1
                    student.assigned_slot = current
                    print(f"  ⚠ No empty slots. Keeping {current}:00")
                break

            elif choice == "C":
                print(f"\n  ✗ {student.name} declined. Cancelling assignment...")
                assigned = [s for s in assigned if s.name != student.name]
                student.assigned_gym  = None
                student.assigned_slot = None

                pref_match, match_idx = relocator.find_preference_match(
                    gym, current, waiting
                )
                if pref_match:
                    pref_match.assigned_gym  = gym
                    pref_match.assigned_slot = current
                    slot_load[gym][current] += 1
                    assigned.append(pref_match)
                    waiting.pop(match_idx)
                    print(f"  → Slot {current}:00 @ Gym {gym} given to "
                          f"{pref_match.name} (was their preferred slot!)")
                elif waiting:
                    first = waiting[0]
                    first.assigned_gym  = gym
                    first.assigned_slot = current
                    slot_load[gym][current] += 1
                    assigned.append(first)
                    waiting.pop(0)
                    print(f"  → Slot {current}:00 @ Gym {gym} given to "
                          f"{first.name} (from waiting list)")
                else:
                    print(f"  → Slot {current}:00 @ Gym {gym} is now free "
                          f"(no one in the waiting queue)")
                break

            else:
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(available):
                        target_slot, holder = available[idx]

                        if holder:
                            print(f"\n  ⚡ Claiming slot {target_slot}:00 from "
                                  f"{holder.name} (Year {holder.year})...")
                            # Free the holder's current slot
                            slot_load[gym][target_slot] -= 1

                            # Auto-reassign holder to nearest available empty slot
                            # so they stay in assigned and get their turn to pick
                            temp_slot = None
                            best_dist = 999
                            for s in ALL_SLOTS:
                                if slot_load[gym][s] < slot_capacity:
                                    dist = abs(s - holder.preferred_slot)
                                    if dist < best_dist:
                                        best_dist = dist
                                        temp_slot = s
                            if temp_slot is not None:
                                holder.assigned_slot = temp_slot
                                slot_load[gym][temp_slot] += 1
                                print(f"  → {holder.name} temporarily moved to "
                                      f"{temp_slot}:00 — will choose a new slot next.")
                            else:
                                # No empty slot in same gym, try other gym
                                other_gym = "B" if gym == "A" else "A"
                                for s in ALL_SLOTS:
                                    if slot_load[other_gym][s] < slot_capacity:
                                        dist = abs(s - holder.preferred_slot)
                                        if dist < best_dist:
                                            best_dist = dist
                                            temp_slot = s
                                if temp_slot is not None:
                                    holder.assigned_gym = other_gym
                                    holder.assigned_slot = temp_slot
                                    slot_load[other_gym][temp_slot] += 1
                                    print(f"  → {holder.name} temporarily moved to "
                                          f"Gym {other_gym} @ {temp_slot}:00 — "
                                          f"will choose a new slot next.")
                                else:
                                    print(f"  → {holder.name} — no empty slots, "
                                          f"added to waiting list.")
                                    assigned = [s for s in assigned
                                                if s.name != holder.name]
                                    holder.assigned_gym  = None
                                    holder.assigned_slot = None
                                    waiting.append(holder)

                            already_processed.discard(holder.name)
                            process_queue.insert(0, holder)

                        student.assigned_slot = target_slot
                        slot_load[gym][target_slot] += 1
                        if target_slot == current:
                            print(f"  ✓ Keeping current slot {current}:00")
                        else:
                            print(f"  ✓ Moved to Gym {gym} @ {target_slot}:00")
                        break
                    else:
                        print(f"  ⚠ Enter 1–{len(available)}, A for auto, or C to cancel.")
                except ValueError:
                    print(f"  ⚠ Enter a number (1–{len(available)}), A for auto, or C to cancel.")

    return assigned, waiting, slot_load

def main():
    print(f"\n{SEP}")
    print("  INTERACTIVE GYM SCHEDULER — with Relocation Consent")
    print(SEP)
    print("  Slots : ", ", ".join(f"{s}:00" for s in VALID_SLOTS))
    print("  Rule  : Students get their preferred slot first.")
    print("          If full, they are relocated to the nearest available slot.")
    print("          Relocated students can accept, pick another slot, or cancel.\n")

    total = read_int("  Number of students: ", minimum=1)
    cap   = read_int("  Slot capacity per gym per hour: ", minimum=1)

    # ── Collect student data ──
    students = []
    for i in range(1, total + 1):
        print(f"\n  ── Student {i} ──")
        gender = read_gender("  Gender (M/F): ")
        year   = read_int(f"  Year (1–{MAX_YEAR}): ", minimum=1, maximum=MAX_YEAR)
        pref   = read_slot("  Preferred slot (6, 7, 8, 16, 17, 18, 19): ")
        travel = read_int("  Travel time (minutes): ", minimum=0)
        students.append(Student(f"S{i:03d}", gender, year, pref, travel))

    # ── Stage 1: Network Flow ──
    max_flow_val, _ = build_and_solve_flow_network(students, cap)
    theoretical_max = len(ALL_SLOTS) * cap * 2
    print(f"\n{SEP}")
    print("  STAGE 1 — Network Flow (Edmonds-Karp)")
    print(SEP)
    print(f"  Max flow: {max_flow_val} / {theoretical_max} "
          f"({round(max_flow_val/theoretical_max*100, 1)}% utilisation)")

    # ── Compute dynamic peak slots from actual demand ──
    peak_slots = compute_peak_slots(students)
    print(f"\n  Peak slots (auto-detected from demand): "
          f"{', '.join(f'{s}:00' for s in peak_slots)}")

    # ── Stage 2: Priority Scheduling ──
    scheduler = GymSlotScheduler({'A': None, 'B': None}, cap, peak_slots=peak_slots)
    for s in students:
        scheduler.submit_request(s)
    assigned, waiting = scheduler.process_all()

    print(f"\n{SEP}")
    print("  STAGE 2 — Priority Scheduling (Min-Heap)")
    print(SEP)
    print(f"  Assigned: {len(assigned)}  |  Waiting: {len(waiting)}")
    slot_load = scheduler.slot_load

    # ── Stage 3: Bin Packing ──
    lii_before = BinPackingBalancer.load_imbalance_index(slot_load)
    balancer = BinPackingBalancer(cap, peak_slots=peak_slots)
    migrated, slot_load = balancer.rebalance(assigned, slot_load)
    lii_after = BinPackingBalancer.load_imbalance_index(slot_load)

    print(f"\n{SEP}")
    print("  STAGE 3 — Bin Packing Load Balancer (FFD)")
    print(SEP)
    if migrated:
        print(f"  Rebalanced: {migrated} student(s) spread from congested to lighter slots")
        print(f"  Load Imbalance Index: {lii_before} → {lii_after} (lower is better)")
    else:
        print(f"  No rebalancing needed — load is already well distributed")
        print(f"  Load Imbalance Index: {lii_after}")

    # ── Show initial assignment ──
    print(f"\n{SEP}")
    print("  INITIAL ASSIGNMENT (before relocation consent)")
    print(SEP)
    for s in assigned:
        print_student_line(s)
    if waiting:
        print(f"\n  Waiting: {', '.join(s.name for s in waiting)}")
    print_slot_loads(slot_load)

    # ── Interactive relocation ──
    assigned, waiting, slot_load = handle_relocations(
        assigned, waiting, slot_load, cap
    )

    # ── Final results ──
    print_full_results(assigned, waiting, slot_load, cap, peak_slots=peak_slots)


if __name__ == "__main__":
    main()
