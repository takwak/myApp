import pandas as pd
from collections import defaultdict
import heapq
import re
import itertools
import time
from datetime import datetime

# Department-specific elective requirements (including General department)
DEPT_ELECTIVE_REQUIREMENTS = {
    5: 7,   # CS
    6: 2,   # IT
    7: 4,   # AI
    8: 1,   # DS
    10: 4,  # IS
    14: 2,  # BIO
    11: 0   # General (no department electives)
}

def clean_course_code(course):
    return str(course).strip().upper().replace(" ", "")

def extract_level(course_code):
    """Extract course level as integer (100, 200, 300, 400)"""
    try:
        course_str = str(course_code).strip().upper().replace(" ", "")
        match = re.search(r'\d{3}', course_str)
        if match:
            return int(match.group()[0]) * 100
        return 100
    except:
        return 100

def get_completed_courses_and_credits(df_tgdidkid, student_id, df_mwad, max_fasel=None, max_zamanno=None):
    credit_map = {}
    for _, row in df_mwad.iterrows():
        course = clean_course_code(row["MadaNo"])
        credit = row["AddWhdat"]
        if pd.isna(credit):
            credit = 3
        credit_map[course] = credit
    credit_map["CS499"] = 4

    student_rows = df_tgdidkid[df_tgdidkid["KidNo"] == student_id]
    
    # Filter by academic year and term if provided
    if max_zamanno is not None and max_fasel is not None:
        mask_prev = student_rows['Zamanno'] < max_zamanno
        mask_current = (student_rows['Zamanno'] == max_zamanno) & (student_rows['FaselNo'] <= max_fasel)
        student_rows = student_rows[mask_prev | mask_current]
    
    completed_courses = set()
    completed_credits = 0

    for val in student_rows["Mongz"].dropna():
        courses = []
        if isinstance(val, str):
            for sep in [",", "-", " "]:
                if sep in val:
                    courses = [clean_course_code(c) for c in val.split(sep) if c.strip()]
                    break
            else:
                courses = [clean_course_code(val)]
        else:
            courses = [clean_course_code(val)]
        
        for course in courses:
            if course not in completed_courses:
                completed_courses.add(course)
                completed_credits += credit_map.get(course, 3)

    return completed_courses, completed_credits

def get_max_credits_per_term(gpa):
    if gpa >= 75:
        return 22
    elif gpa >= 65:
        return 19
    elif gpa >= 50:
        return 17
    else:
        return 15

def has_level_conflict(courses, credit_map):
    """Check if courses violate level constraints (100 with 300/400, 200 with 400)"""
    levels = set()
    for course in courses:
        if course == "ELECTIVE":
            continue
        level = extract_level(course)
        levels.add(level)
    
    # Check for conflicts
    if 100 in levels:
        if 300 in levels or 400 in levels:
            return True
    if 200 in levels:
        if 400 in levels:
            return True
    return False

def topological_graduation_path(dept, current_whdat, completed, df_kesmmwad, df_mwad, cpn_dict, gpa, target_credits=134, timeout_seconds=30):
    """Topological sorting fallback with level constraints and timing"""
    start_time = time.time()
    print("🔁 Using topological sorting fallback with level constraints...")
    
    # Create clean credit map
    credit_map = {}
    for _, row in df_mwad.iterrows():
        course = clean_course_code(row["MadaNo"])
        credit = row["AddWhdat"]
        if pd.isna(credit):
            credit = 3
        credit_map[course] = credit
    credit_map["CS499"] = 4
    credit_map["ELECTIVE"] = 3

    max_credits_per_term = get_max_credits_per_term(gpa)
    dept_elective_count = DEPT_ELECTIVE_REQUIREMENTS.get(dept, 0)
    
    # Identify core courses
    core_set = set()
    if dept in cpn_dict:
        G = cpn_dict[dept]
        for course in G.nodes():
            clean_course = clean_course_code(course)
            if clean_course != "VIRTUAL_START":
                core_set.add(clean_course)
    core_set.add("CS499")
    
    # Identify general education courses
    general_courses = set(df_kesmmwad[df_kesmmwad["KesmNo"] == 11]["MadaNo"].dropna().apply(clean_course_code))
    
    # Count completed elective courses
    elective_courses_completed = 0
    for c in completed:
        c_clean = clean_course_code(c)
        if c_clean not in core_set and c_clean not in general_courses:
            elective_courses_completed += 1
    
    # Calculate remaining elective requirements
    elective_courses_needed = max(0, dept_elective_count - elective_courses_completed)
    
    # Required core courses (not completed)
    required_courses = [c for c in core_set if clean_course_code(c) not in completed]
    
    # Build prerequisite map
    prerequisites_map = defaultdict(set)
    criticality_map = defaultdict(int)
    if dept in cpn_dict:
        G = cpn_dict[dept]
        for course in G.nodes():
            clean_course = clean_course_code(course)
            if clean_course == "VIRTUAL_START":
                continue
            for pred in G.predecessors(course):
                clean_pred = clean_course_code(pred)
                if clean_pred != "VIRTUAL_START":
                    prerequisites_map[clean_course].add(clean_pred)
                    criticality_map[clean_pred] += 1
    
    # SPECIAL HANDLING FOR CS499
    prerequisites_map["CS499"] = set()
    
    # Create in-degree map for topological sort
    in_degree = {}
    graph = defaultdict(list)
    for course in required_courses:
        in_degree[course] = 0
        
    for course in required_courses:
        for prereq in prerequisites_map[course]:
            if prereq in required_courses:
                graph[prereq].append(course)
                in_degree[course] += 1
    
    # Find starting nodes (courses with no prerequisites)
    queue = [course for course in required_courses if in_degree[course] == 0]
    heapq.heapify(queue)
    
    topological_order = []
    while queue:
        # Check timeout
        if time.time() - start_time > timeout_seconds:
            print(f"⏱️ Topological sorting timed out after {timeout_seconds} seconds")
            return [], current_whdat
        
        course = heapq.heappop(queue)
        topological_order.append(course)
        for neighbor in graph[course]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                heapq.heappush(queue, neighbor)
    
    # Check if all courses were processed
    if len(topological_order) != len(required_courses):
        print("⚠️ Cycle detected in prerequisite graph - cannot use topological sort")
        return [], current_whdat
    
    # Plan terms using topological order with level constraints
    plan = []
    current_term = []
    current_levels = set()
    current_credits = 0
    cumulative_credits = current_whdat
    elective_count = elective_courses_completed
    
    # Add CS499 only when 100 credits are reached
    cs499_added = False
    cs499_index = None
    if "CS499" in topological_order:
        cs499_index = topological_order.index("CS499")
    
    # Create a list for remaining courses
    remaining_courses = topological_order.copy()
    
    while remaining_courses:
        # Check timeout
        if time.time() - start_time > timeout_seconds:
            print(f"⏱️ Topological sorting timed out after {timeout_seconds} seconds")
            return plan, cumulative_credits
        
        # Check if we've reached target credits for specialization
        if cumulative_credits >= target_credits and elective_count >= dept_elective_count:
            break
            
        added_course = False
        
        # Try to add courses that don't cause conflicts
        for i in range(len(remaining_courses)):
            course = remaining_courses[i]
            # Skip CS499 if not enough credits
            if course == "CS499" and cumulative_credits + current_credits < 100:
                continue
                
            course_credit = credit_map.get(course, 3)
            course_level = extract_level(course)
            
            # Check if adding this course would exceed credit limit
            if current_credits + course_credit > max_credits_per_term:
                continue
                
            # Check level conflicts
            conflict = False
            if course_level == 100:
                if 300 in current_levels or 400 in current_levels:
                    conflict = True
            elif course_level == 200:
                if 400 in current_levels:
                    conflict = True
            elif course_level == 300:
                if 100 in current_levels:
                    conflict = True
            elif course_level == 400:
                if 100 in current_levels or 200 in current_levels:
                    conflict = True
            
            if conflict:
                continue
            
            # Add the course to current term
            current_term.append(course)
            current_levels.add(course_level)
            current_credits += course_credit
            remaining_courses.pop(i)
            added_course = True
            break
        
        # If no course was added, finalize the term
        if not added_course:
            # Add electives if needed and there's space
            while elective_count < dept_elective_count and current_credits + 3 <= max_credits_per_term:
                current_term.append("ELECTIVE")
                current_credits += 3
                elective_count += 1
            
            if current_term:
                plan.append(current_term)
                cumulative_credits += current_credits
                current_term = []
                current_levels = set()
                current_credits = 0
            else:
                # Break if we can't add anything
                break
    
    # Add last term if not empty
    if current_term:
        # Add electives if needed and there's space
        while elective_count < dept_elective_count and current_credits + 3 <= max_credits_per_term:
            current_term.append("ELECTIVE")
            current_credits += 3
            elective_count += 1
        
        plan.append(current_term)
        cumulative_credits += current_credits
    
    # Add CS499 if not added yet
    if "CS499" in topological_order and "CS499" not in [c for term in plan for c in term]:
        # Find best term to add CS499
        for i, term in enumerate(plan):
            term_credits = sum(credit_map.get(c, 3) for c in term if c != "ELECTIVE")
            term_credits += 3 * sum(1 for c in term if c == "ELECTIVE")
            if term_credits + 4 <= max_credits_per_term:
                term.append("CS499")
                cumulative_credits += 4
                break
        else:
            # Add new term for CS499
            plan.append(["CS499"])
            cumulative_credits += 4
    
    # Add additional terms for electives if needed
    while elective_count < dept_elective_count:
        term = []
        term_credits = 0
        # Add as many electives as possible in one term
        while elective_count < dept_elective_count and term_credits + 3 <= max_credits_per_term:
            term.append("ELECTIVE")
            term_credits += 3
            elective_count += 1
        plan.append(term)
        cumulative_credits += term_credits
    
    return plan, cumulative_credits

def dijkstra_graduation_path(dept, current_whdat, completed, df_kesmmwad, df_mwad, cpn_dict, gpa, target_credits=134, timeout_seconds=120):
    """Dijkstra-based pathfinding with timing and credit targets"""
    start_time = time.time()
    
    # Create clean credit map
    credit_map = {}
    for _, row in df_mwad.iterrows():
        course = clean_course_code(row["MadaNo"])
        credit = row["AddWhdat"]
        if pd.isna(credit):
            credit = 3
        credit_map[course] = credit
    credit_map["CS499"] = 4
    credit_map["ELECTIVE"] = 3

    max_credits_per_term = get_max_credits_per_term(gpa)
    dept_elective_count = DEPT_ELECTIVE_REQUIREMENTS.get(dept, 0)
    
    # Identify core courses
    core_set = set()
    if dept in cpn_dict:
        G = cpn_dict[dept]
        for course in G.nodes():
            clean_course = clean_course_code(course)
            if clean_course != "VIRTUAL_START":
                core_set.add(clean_course)
    core_set.add("CS499")
    
    # Identify general education courses
    general_courses = set(df_kesmmwad[df_kesmmwad["KesmNo"] == 11]["MadaNo"].dropna().apply(clean_course_code))
    
    # Count completed elective courses
    elective_courses_completed = 0
    for c in completed:
        c_clean = clean_course_code(c)
        if c_clean not in core_set and c_clean not in general_courses:
            elective_courses_completed += 1
    
    # Calculate remaining elective requirements
    elective_courses_needed = max(0, dept_elective_count - elective_courses_completed)
    
    # Required core courses (not completed)
    required_courses = [c for c in core_set if clean_course_code(c) not in completed]
    
    # Build prerequisite map
    prerequisites_map = defaultdict(set)
    criticality_map = defaultdict(int)
    if dept in cpn_dict:
        G = cpn_dict[dept]
        for course in G.nodes():
            clean_course = clean_course_code(course)
            if clean_course == "VIRTUAL_START":
                continue
            for pred in G.predecessors(course):
                clean_pred = clean_course_code(pred)
                if clean_pred != "VIRTUAL_START":
                    prerequisites_map[clean_course].add(clean_pred)
                    criticality_map[clean_pred] += 1
    
    # SPECIAL HANDLING FOR CS499
    prerequisites_map["CS499"] = set()
    
    # Print requirements
    print(f"- Core courses needed: {', '.join(required_courses) or 'None'}")
    print(f"- Electives needed: {elective_courses_needed}")
    if target_credits == 134:
        print(f"- CS499 requires 100+ credits (no course prerequisites)")
    
    # Print missing prerequisites
    print("\n🔍 Missing Prerequisites Analysis:")
    any_missing = False
    for course in required_courses:
        clean_course = clean_course_code(course)
        if clean_course == "CS499":
            continue
        missing = [p for p in prerequisites_map.get(clean_course, set()) if p not in completed]
        if missing:
            print(f"- {clean_course}: missing {', '.join(missing)}")
            any_missing = True
    if not any_missing:
        print("- All prerequisites satisfied for available courses")
    
    # Create bitmask mapping for core courses
    core_course_list = sorted(required_courses)
    course_to_index = {course: idx for idx, course in enumerate(core_course_list)}
    
    # Dijkstra setup with optimized state
    start_mask = 0
    start_elective_count = elective_courses_completed
    start_state = (start_mask, start_elective_count)
    priority_queue = [(0, 0, start_state, [])]  # (priority, term_count, state, path)
    best_terms = defaultdict(lambda: float('inf'))
    
    # Track best solution
    min_terms = float('inf')
    best_plan = None
    best_final_credits = 0
    states_processed = 0
    max_states = 500000
    last_progress_time = time.time()
    
    while priority_queue and states_processed < max_states:
        # Check timeout
        if time.time() - start_time > timeout_seconds:
            print(f"⏱️ Dijkstra computation timed out after {timeout_seconds} seconds")
            break
            
        states_processed += 1
        _, term_count, state, path = heapq.heappop(priority_queue)
        core_mask, elective_count = state
        
        # Calculate current credits
        core_credits = 0
        for i, course in enumerate(core_course_list):
            if core_mask & (1 << i):
                core_credits += credit_map.get(course, 3)
        cumulative_credits = current_whdat + core_credits + 3 * (elective_count - elective_courses_completed)
        
        # Check graduation
        if cumulative_credits >= target_credits and elective_count >= dept_elective_count:
            if term_count < min_terms:
                min_terms = term_count
                best_plan = path
                best_final_credits = cumulative_credits
            continue
        
        # Skip if we've found a better path to this state
        if term_count >= best_terms[state]:
            continue
        best_terms[state] = term_count
        
        # Get available core courses
        available_courses = []
        for i, course in enumerate(core_course_list):
            if core_mask & (1 << i):
                continue  # Already taken
                
            # CS499 credit requirement
            if course == "CS499" and cumulative_credits < 100 and target_credits == 134:
                continue
                
            # Check prerequisites
            prereq_met = True
            for p in prerequisites_map.get(course, set()):
                # Handle non-core prerequisites in completed set
                if p in completed or p in core_set:
                    if p in core_set and p in core_course_list:
                        idx = course_to_index[p]
                        if not (core_mask & (1 << idx)):
                            prereq_met = False
                else:
                    prereq_met = False
            if prereq_met:
                available_courses.append(course)
        
        # Sort by criticality (courses unlocking more courses first)
        available_courses.sort(key=lambda x: criticality_map.get(x, 0), reverse=True)
        
        # Generate course combinations - MAXIMIZE CREDIT UTILIZATION
        valid_combinations = []
        n = len(available_courses)
        counter = 0
        max_credits_found = 0
        
        # Consider combinations from largest to smallest
        max_courses_per_term = min(7, max_credits_per_term // 3)
        for r in range(min(max_courses_per_term, n), 0, -1):
            if counter >= 1000:  # Limit combinations per state
                break
            for combo in itertools.combinations(available_courses, r):
                if counter >= 1000:
                    break
                combo = list(combo)
                credits = sum(credit_map.get(c, 3) for c in combo)
                
                # Skip if exceeds max credits
                if credits > max_credits_per_term:
                    continue
                    
                # Check level conflicts
                if has_level_conflict(combo, credit_map):
                    continue
                    
                # Track max credits found
                if credits > max_credits_found:
                    max_credits_found = credits
                    
                valid_combinations.append((combo, credits))
                counter += 1
                
                # Early termination if max credits achieved
                if credits == max_credits_per_term:
                    counter = 1000
                    break
        
        # Sort combinations by credit utilization (descending)
        valid_combinations.sort(key=lambda x: x[1], reverse=True)
        
        # Add elective options with MAXIMUM UTILIZATION
        if elective_count < dept_elective_count:
            # Elective-only options
            max_electives = min(
                dept_elective_count - elective_count,
                max_credits_per_term // 3
            )
            # Add options with maximum possible electives
            for i in range(max_electives, 0, -1):
                valid_combinations.append((["ELECTIVE"] * i, i * 3))
            
            # Combinations with core + electives (FILL TO MAX CREDITS)
            for core_combo, core_credits in valid_combinations[:]:
                remaining = max_credits_per_term - core_credits
                max_add = min(
                    dept_elective_count - elective_count,
                    remaining // 3
                )
                if max_add > 0:
                    # Add option with MAXIMUM electives
                    new_combo = core_combo + ["ELECTIVE"] * max_add
                    new_credits = core_credits + max_add * 3
                    # Check level conflicts for core + electives
                    # (Electives are assumed to avoid conflicts)
                    valid_combinations.append((new_combo, new_credits))
        
        # Generate next states
        for courses, term_credits in valid_combinations:
            # Skip underload unless graduating
            will_graduate = (
                cumulative_credits + term_credits >= target_credits and 
                elective_count + courses.count("ELECTIVE") >= dept_elective_count
            )
            if term_credits < 14 and not will_graduate:
                continue
                
            # Update core mask
            new_mask = core_mask
            new_elective = elective_count
            for course in courses:
                if course == "ELECTIVE":
                    new_elective += 1
                else:
                    idx = course_to_index[course]
                    new_mask |= (1 << idx)
            
            new_state = (new_mask, new_elective)
            new_path = path + [courses]
            
            # Calculate priority (term count + remaining courses heuristic)
            remaining_courses = len(core_course_list) - bin(new_mask).count("1")
            priority = term_count + 1 + (remaining_courses / 4)
            
            # Only add if better than existing path
            if term_count + 1 < best_terms.get(new_state, float('inf')):
                heapq.heappush(priority_queue, (priority, term_count + 1, new_state, new_path))
    
    if best_plan:
        return best_plan, best_final_credits
    
    print(f"⚠️ Dijkstra found no path after processing {states_processed} states")
    
    # Try topological fallback with same parameters
    print("🔁 Attempting topological fallback with level constraints...")
    return topological_graduation_path(
        dept, current_whdat, completed, df_kesmmwad, df_mwad, cpn_dict, gpa,
        target_credits=target_credits, timeout_seconds=30
    )

def recommend_fastest_path(student_id, df_tgdidkid, df_mwad, df_kesmmwad, cpn_dict, target_dept=5):
    print("🚀 Starting Optimized Graduation Planner with Level Constraints...")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    student_data = df_tgdidkid[df_tgdidkid["KidNo"] == student_id]
    
    if student_data.empty:
        print(f"⚠️ Student {student_id} not found in records")
        return []
    
    # Get the two most recent terms
    sorted_terms = student_data.sort_values(by=['Zamanno', 'FaselNo'], ascending=[False, False])
    if len(sorted_terms) >= 2:
        # Use the second largest FaselNo for student status
        last_completed_term = sorted_terms.iloc[1]
        # Get the largest FaselNo (current term)
        last_term = sorted_terms.iloc[0]
    else:
        # If only one term, use it for status
        last_completed_term = sorted_terms.iloc[0]
        last_term = last_completed_term
    
    current_gpa = float(last_completed_term["Average"]) if pd.notna(last_completed_term["Average"]) else 75.0
    current_fasel = int(last_completed_term["FaselNo"])
    zamano = int(last_completed_term["Zamanno"])
    current_zamano = int(last_term["Zamanno"])
    next_term_start = int(last_term["FaselNo"])   # Start from next term
    
    if zamano < 46:
        print(f"⚠️ Student {student_id} is not from current academic year (Zamano={zamano})")
        return []
    
    # Get completed courses and credits up to last completed term
    completed, current_whdat = get_completed_courses_and_credits(
        df_tgdidkid, student_id, df_mwad, 
        max_fasel=current_fasel, 
        max_zamanno=zamano
    )
    
    dept = int(last_completed_term["KesmNo"])
    
    # Build department-specific core set
    dept_core_set = set()
    if dept != 11 and dept in cpn_dict:  # Specialized department
        G = cpn_dict[dept]
        for course in G.nodes():
            clean_course = clean_course_code(course)
            if clean_course != "VIRTUAL_START":
                dept_core_set.add(clean_course)
        dept_core_set.add("CS499")
    
    # Identify general education courses
    general_courses = set(df_kesmmwad[df_kesmmwad["KesmNo"] == 11]["MadaNo"].dropna().apply(clean_course_code))
    
    # Count completed elective courses
    elective_courses_completed = 0
    for c in completed:
        c_clean = clean_course_code(c)
        if c_clean not in dept_core_set and c_clean not in general_courses:
            elective_courses_completed += 1
    
    print(f"\n📊 Student Status:")
    print(f"- Department: {dept} ({'General' if dept == 11 else 'Specialized'})")
    print(f"- Completed Credits: {current_whdat}")
    print(f"- Current GPA: {current_gpa:.2f}")
    print(f"- Last Completed Term: {current_fasel}")
    print(f"- Current Term: {last_term['FaselNo']}")
    print(f"- Academic Year: {zamano}")
    print(f"- Completed Electives: {elective_courses_completed}")
    
    if dept != 11:
        dept_elective_count = DEPT_ELECTIVE_REQUIREMENTS.get(dept, 0)
        print(f"- Department Elective Requirement: {dept_elective_count} (remaining: {max(0, dept_elective_count - elective_courses_completed)})")
    
    if current_whdat >= 134:
        print("\n🎉 Congratulations! You have already graduated.")
        return []
    
    # Specialized department students
    if dept != 11:
        print("\n📚 Finding Fastest Graduation Path...")
        start_time = time.time()
        plan, total_credits = dijkstra_graduation_path(
            dept, current_whdat, completed, df_kesmmwad, df_mwad, cpn_dict, current_gpa
        )
        dijkstra_time = time.time() - start_time
        print(f"⏱️ Total planning time: {dijkstra_time:.2f} seconds")
        
        if not plan:
            print("⚠️ No valid graduation path found")
            return []
        
        cumulative = current_whdat
        for i, term in enumerate(plan, start=next_term_start):
            term_credits = 0
            term_courses = []
            for course in term:
                if course == "ELECTIVE":
                    term_credits += 3
                    term_courses.append("ELECTIVE")
                elif course == "CS499":
                    term_credits += 4
                    term_courses.append("CS499")
                else:
                    credit = df_mwad.set_index("MadaNo")["AddWhdat"].get(course, 3)
                    term_credits += credit
                    term_courses.append(course)
            cumulative += term_credits
            print(f"\nTerm {i}:")
            print(f"- Courses: {', '.join(term_courses)}")
            print(f"- Credits: {term_credits}")
            print(f"- Cumulative Credits: {cumulative}")
            if "CS499" in term:
                print("- Note: 🎓 Graduation Project (CS499) included")
        print(f"\n🎓 Total projected credits: {cumulative}/134")
        print(f"⏱️ Fastest path in {len(plan)} terms")
        return plan
    
    # General department students
    if current_whdat < 45:
        print("\n📚 General Department: Path to 45 Credits...")
        start_time = time.time()
        plan_general, cumulative = dijkstra_graduation_path(
            11, current_whdat, completed, df_kesmmwad, df_mwad, cpn_dict, current_gpa,
            target_credits=45, timeout_seconds=30
        )
        dijkstra_time = time.time() - start_time
        print(f"⏱️ Total planning time: {dijkstra_time:.2f} seconds")
        
        if not plan_general:
            print("⚠️ No valid path to 45 credits found")
            return []
        
        # Combine completed courses
        new_completed = set(completed)
        for term in plan_general:
            for course in term:
                new_completed.add(clean_course_code(course))
        
        # Print general plan
        cumulative_credits = current_whdat
        for i, term in enumerate(plan_general, start=next_term_start):
            term_credits = 0
            term_courses = []
            for course in term:
                if course == "ELECTIVE":
                    term_credits += 3
                    term_courses.append("ELECTIVE")
                else:
                    credit = df_mwad.set_index("MadaNo")["AddWhdat"].get(course, 3)
                    term_credits += credit
                    term_courses.append(course)
            cumulative_credits += term_credits
            print(f"\nTerm {i}:")
            print(f"- Courses: {', '.join(term_courses)}")
            print(f"- Credits: {term_credits}")
            print(f"- Cumulative Credits: {cumulative_credits}")
        
        # Update cumulative credits
        cumulative = cumulative_credits
        print(f"\n🎉 Reached {cumulative} credits - ready to specialize!")
    else:
        print(f"\n🎉 Already reached {current_whdat} credits - ready to specialize!")
        new_completed = completed
        cumulative = current_whdat
        plan_general = []
    
    # Use target department if provided
    if target_dept is None:
        print("\n⚠️ Target department not specified. Using Computer Science (5) as default.")
        target_dept = 5
    
    print(f"\n📚 Specialized Department ({target_dept}): Fastest Graduation Path...")
    start_time = time.time()
    plan_specialized, total_credits = dijkstra_graduation_path(
        target_dept,
        cumulative,
        new_completed,
        df_kesmmwad,
        df_mwad,
        cpn_dict,
        current_gpa
    )
    dijkstra_time = time.time() - start_time
    print(f"⏱️ Total planning time: {dijkstra_time:.2f} seconds")
    
    if not plan_specialized:
        print("⚠️ No valid graduation path found after specialization")
        return plan_general if 'plan_general' in locals() else []
    
    full_plan = (plan_general if 'plan_general' in locals() else []) + plan_specialized
    cumulative_credits = cumulative
    
    start_index = len(plan_general) + next_term_start if 'plan_general' in locals() else next_term_start
    for i, term in enumerate(plan_specialized, start=start_index):
        term_credits = 0
        term_courses = []
        for course in term:
            if course == "ELECTIVE":
                term_credits += 3
                term_courses.append("ELECTIVE")
            elif course == "CS499":
                term_credits += 4
                term_courses.append("CS499")
            else:
                credit = df_mwad.set_index("MadaNo")["AddWhdat"].get(course, 3)
                term_credits += credit
                term_courses.append(course)
        cumulative_credits += term_credits
        print(f"\nTerm {i}:")
        print(f"- Courses: {', '.join(term_courses)}")
        print(f"- Credits: {term_credits}")
        print(f"- Cumulative Credits: {cumulative_credits}")
        if "CS499" in term:
            print("- Note: 🎓 Graduation Project (CS499) included")
    
    print(f"\n🎓 Total projected credits: {cumulative_credits}/134")
    if 'plan_general' in locals():
        print(f"⏱️ Full path in {len(full_plan)} terms ({len(plan_general)} general + {len(plan_specialized)} specialized)")
    else:
        print(f"⏱️ Specialized path in {len(plan_specialized)} terms")
    return full_plan
