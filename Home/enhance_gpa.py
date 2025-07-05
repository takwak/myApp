import pandas as pd
from collections import defaultdict, deque
import numpy as np
from itertools import islice
import warnings
import math
import re  # Added for extract_level function

# Department-specific elective requirements
DEPT_ELECTIVE_REQUIREMENTS = {
    5: 7,   # IN
    6: 2,   # CS
    7: 4,   # SE
    8: 1,   # CN
    10: 4,  # MM
    14: 2   # IS
}

def has_level_conflict_in_term(term_courses, level_map):
    levels_present = set()
    
    for course in term_courses:
        # Ignore electives and graduation project
        if course in ["ELECTIVE", "CS499"]:
            continue
            
        # Get course level
        level = level_map.get(course, extract_level(course))
        levels_present.add(level)
    
    # Check for conflicts
    if 100 in levels_present:
        if 300 in levels_present or 400 in levels_present:
            return True
    if 200 in levels_present:
        if 400 in levels_present:
            return True
    return False

# Clean course code function
def clean_course_code(course):
    return str(course).strip().upper().replace(" ", "")

# Extract course level
def extract_level(course_code):
    try:
        course_str = str(course_code).strip().upper().replace(" ", "")
        match = re.search(r'\d', course_str)
        if match:
            return int(match.group()) * 100
        return 100
    except:
        return 100

# Get completed courses and credits
def get_completed_courses_and_credits(df_tgdidkid, student_id, df_mwad, max_fasel=None):
    credit_map = {}
    for _, row in df_mwad.iterrows():
        course = clean_course_code(row["MadaNo"])
        credit = row["AddWhdat"]
        if pd.isna(credit):
            credit = 3
        credit_map[course] = credit
    credit_map["CS499"] = 4

    student_rows = df_tgdidkid[df_tgdidkid["KidNo"] == student_id]
    if max_fasel is not None:
        student_rows = student_rows[student_rows["FaselNo"] <= max_fasel]
    
    completed_courses = set()
    completed_credits = 0

    for _, row in student_rows.iterrows():
        val = row["Mongz"]
        if pd.isna(val):
            continue
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

# Calculate term credits
def calculate_term_credits(term, credit_map):
    total = 0
    for course in term:
        if course == "ELECTIVE":
            total += 3
        elif course == "CS499":
            total += 4
        else:
            total += credit_map.get(course, 3)
    return total

# Get max credits based on GPA
def get_max_credits_per_term(gpa):
    if gpa >= 75:
        return 22
    elif gpa >= 65:
        return 19
    elif gpa >= 50:
        return 17
    else:
        return 15

# Predict term GPA using actual model features
def predict_term_gpa(term_courses, current_gpa, cumulative_credits, dept, 
                     model, tfidf, model_features_list, credit_map):
    """Predict GPA impact of a full term using actual model features"""
    # Create term string representation
    term_str = '-'.join(sorted([clean_course_code(c) for c in term_courses]))
    
    # Transform term string using TF-IDF
    tfidf_vector = tfidf.transform([term_str])
    tfidf_df_term = pd.DataFrame(
        tfidf_vector.toarray(), 
        columns=tfidf.get_feature_names_out()
    )
    
    # Calculate term credits
    term_credits = calculate_term_credits(term_courses, credit_map)
    
    # Create feature vector
    features = {
        'PreviousGPA': current_gpa,
        'CourseCount': len(term_courses),
        'TotalWhdat': term_credits,
        'WhdatMongz': cumulative_credits,
        'KesmNo': dept
    }
    
    # Create feature DataFrame
    feature_df = pd.DataFrame([features])
    feature_df = pd.concat([feature_df, tfidf_df_term], axis=1)
    
    # Ensure correct feature columns
    for col in model_features_list:
        if col not in feature_df.columns:
            feature_df[col] = 0
    feature_df = feature_df[model_features_list]
    
    # Predict term GPA
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        term_gpa = model.predict(feature_df)[0]
    
    return term_gpa, term_credits

# Topological sort for course dependencies
def topological_sort(G):
    in_degree = {node: 0 for node in G.nodes()}
    for node in G.nodes():
        for successor in G.successors(node):
            in_degree[successor] += 1
    
    queue = deque([node for node in G.nodes() if in_degree[node] == 0])
    sorted_nodes = []
    
    while queue:
        node = queue.popleft()
        sorted_nodes.append(node)
        
        for successor in G.successors(node):
            in_degree[successor] -= 1
            if in_degree[successor] == 0:
                queue.append(successor)
    
    if len(sorted_nodes) != len(G.nodes()):
        print("Warning: Graph has cycles, topological sort may be incomplete")
    
    return sorted_nodes

# Find optimal path through courses
def find_optimal_path(G, completed, current_gpa, cumulative_credits, dept, model, model_features_list):
    try:
        sorted_courses = topological_sort(G)
        sorted_courses = [c for c in sorted_courses if c not in ['VIRTUAL_START', 'VIRTUAL_END'] and c not in completed]
        return sorted_courses, current_gpa
    except:
        return [n for n in G.nodes() if n not in ['VIRTUAL_START', 'VIRTUAL_END'] and n not in completed], current_gpa

def all_topological_sorts(graph):
    """Yield all topological orderings of a DAG"""
    def helper(G, visited, stack, in_degree):
        flag = False
        for v in G.nodes():
            if not visited[v] and in_degree[v] == 0:
                visited[v] = True
                stack.append(v)
                for n in G.successors(v):
                    in_degree[n] -= 1

                yield from helper(G, visited, stack, in_degree)

                visited[v] = False
                stack.pop()
                for n in G.successors(v):
                    in_degree[n] += 1
                flag = True
        if not flag:
            yield list(stack)
    visited = {v: False for v in graph.nodes()}
    in_degree = {v: 0 for v in graph.nodes()}
    for u in graph.nodes():
        for v in graph.successors(u):
            in_degree[v] += 1
    yield from helper(graph, visited, [], in_degree)

def ensure_min_credits(term_courses, term_credits, elective_courses_needed, 
                       level_map, max_credits_per_term, min_required):
    """Add electives to meet minimum credit requirement if possible"""
    added = 0
    # Only try to add if we're below minimum and have electives available
    if term_credits < min_required and elective_courses_needed > 0:
        # Calculate how many electives we can add
        needed_credits = min_required - term_credits
        max_electives = min(
            elective_courses_needed,
            needed_credits // 3,
            (max_credits_per_term - term_credits) // 3
        )
        
        # Try to add each elective
        for _ in range(max_electives):
            candidate_term = term_courses + ["ELECTIVE"]
            if has_level_conflict_in_term(candidate_term, level_map):
                break
            term_courses = candidate_term
            term_credits += 3
            added += 1
            
    return term_courses, term_credits, added

def schedule_terms_from_path(
    path, model, tfidf, model_features_list, credit_map, level_map, get_max_credits_per_term,
    prev_gpa, cumulative_credits, dept, elective_courses_needed
):
    """Given a path (course sequence), simulate scheduling into terms, predict GPA for each term, and return final projected GPA."""
    scheduled = []
    current_gpa = prev_gpa
    scheduled_courses = set()
    i = 0
    term_count = 0
    cumulative = cumulative_credits
    remaining_electives = elective_courses_needed
    
    while i < len(path):
        term_courses = []
        term_credits = 0
        max_term_credits = get_max_credits_per_term(current_gpa)
        remaining_total = 134 - cumulative
        
        # Calculate minimum credits for this term (14 unless last term)
        min_required = min(14, remaining_total) if remaining_total > 0 else 0
        
        # Fill term up to max credits
        while i < len(path):
            course = path[i]
            if course in scheduled_courses or course in ("VIRTUAL_START", "VIRTUAL_END"):
                i += 1
                continue
                
            course_credits = credit_map.get(course, 3)
            
            # Check if adding course would exceed credit limit
            if term_credits + course_credits > max_term_credits:
                break
                
            # Create candidate term to check for level conflict
            candidate_term = term_courses + [course]
            if has_level_conflict_in_term(candidate_term, level_map):
                break
                
            term_courses.append(course)
            term_credits += course_credits
            scheduled_courses.add(course)
            i += 1
        
        # Add electives to meet minimum credit requirement if needed
        if term_credits < min_required:
            term_courses, term_credits, added = ensure_min_credits(
                term_courses, term_credits, remaining_electives,
                level_map, max_term_credits, min_required
            )
            remaining_electives -= added
            
        if not term_courses:
            break
            
        # Predict GPA for this term
        new_gpa, actual_credits = predict_term_gpa(
            term_courses, current_gpa, cumulative, dept,
            model, tfidf, model_features_list, credit_map
        )
        
        cumulative += actual_credits
        term_count += 1
        scheduled.append((term_courses, new_gpa, actual_credits))
        current_gpa = new_gpa
    
    return scheduled, current_gpa, cumulative, remaining_electives

def find_best_gpa_path_for_student(
    dept, cpn_dict, model, tfidf, model_features_list, credit_map, level_map, get_max_credits_per_term,
    prev_gpa, initial_credits, completed, elective_courses_needed, max_paths=1000
):
    """Enumerate valid paths and pick the path with the highest final GPA."""
    if dept not in cpn_dict:
        return [], prev_gpa, initial_credits, elective_courses_needed
        
    G = cpn_dict[dept].copy()
    
    # Remove completed courses and virtual nodes
    for node in list(G.nodes()):
        if node in completed or node in ['VIRTUAL_START', 'VIRTUAL_END']:
            G.remove_node(node)
    
    # If no courses left, return empty schedule
    if len(G.nodes()) == 0:
        return [], prev_gpa, initial_credits, elective_courses_needed
    
    best_gpa = -np.inf
    best_schedule = None
    best_cumulative = initial_credits
    best_remaining_electives = elective_courses_needed
    path_count = 0
    
    # Generate and evaluate paths
    for path in islice(all_topological_sorts(G), max_paths):
        path_count += 1
        schedule, gpa, cumulative, remaining_electives = schedule_terms_from_path(
            path, model, tfidf, model_features_list, credit_map, level_map,
            get_max_credits_per_term, prev_gpa, initial_credits, dept, elective_courses_needed
        )
        
        if gpa > best_gpa:
            best_gpa = gpa
            best_schedule = schedule
            best_cumulative = cumulative
            best_remaining_electives = remaining_electives
    
    print(f"Evaluated {path_count} paths, best GPA: {best_gpa:.2f}")
    return best_schedule, best_gpa, best_cumulative, best_remaining_electives

# GPA-OPTIMIZED COURSE SELECTION
def optimize_term_courses(available_courses, min_credits, max_credits, current_gpa, cumulative_credits, dept, 
                          model, tfidf, model_features_list, credit_map, level_map):
    """Select courses that maximize GPA using knapsack approach with level constraints"""
    # Predict grades for each course individually
    course_grades = {}
    for course in available_courses:
        term_gpa, _ = predict_term_gpa(
            [course], current_gpa, cumulative_credits, dept,
            model, tfidf, model_features_list, credit_map
        )
        course_grades[course] = term_gpa
    
    # Calculate grade points (grade * credits)
    course_data = []
    for course in available_courses:
        credit = credit_map.get(course, 3)
        grade_points = course_grades[course] * credit
        course_data.append((course, grade_points, credit))
    
    # Sort by grade points (descending)
    course_data.sort(key=lambda x: x[1], reverse=True)
    
    # Greedy selection to maximize grade points within credit limits
    selected = []
    total_credits = 0
    total_grade_points = 0
    
    # First pass: select high-value courses
    for course, grade_points, credit in course_data:
        # Create candidate term to check for level conflict
        candidate_term = selected + [course]
        if has_level_conflict_in_term(candidate_term, level_map):
            continue
            
        if total_credits + credit <= max_credits:
            selected.append(course)
            total_credits += credit
            total_grade_points += grade_points
    
    # If below minimum credits, add courses with highest predicted grades
    if total_credits < min_credits:
        remaining = [c for c in available_courses if c not in selected]
        remaining.sort(key=lambda c: course_grades.get(c, 0), reverse=True)
        
        for course in remaining:
            # Create candidate term to check for level conflict
            candidate_term = selected + [course]
            if has_level_conflict_in_term(candidate_term, level_map):
                continue
                
            credit = credit_map.get(course, 3)
            if total_credits + credit <= max_credits:
                selected.append(course)
                total_credits += credit
                if total_credits >= min_credits:
                    break
    
    return selected, total_credits

# Main recommendation function
def recommend_specialized_path(dept, current_whdat, completed, df_kesmmwad, df_mwad, cpn_dict, 
                               gpa, model, tfidf, credit_map, level_map, model_features_list, start_term=1):
    # Create clean credit map
    for _, row in df_mwad.iterrows():
        course = clean_course_code(row["MadaNo"])
        if course not in credit_map:
            credit = row["AddWhdat"]
            credit_map[course] = 3 if pd.isna(credit) else credit
    credit_map["CS499"] = 4
    credit_map["ELECTIVE"] = 3

    # Build level map for all courses
    for course in credit_map:
        if course not in level_map:
            level_map[course] = extract_level(course)
    level_map["ELECTIVE"] = 0
    level_map["CS499"] = 0

    current_gpa = gpa
    max_credits_per_term = get_max_credits_per_term(current_gpa)
    print(f"📊 Credit Limit: {max_credits_per_term} credits/term (Starting GPA: {current_gpa:.2f})")

    dept_elective_count = DEPT_ELECTIVE_REQUIREMENTS.get(dept, 0)
    print(f"📊 Department {dept} requires {dept_elective_count} elective courses")

    # Identify core courses
    core_set = set()
    if dept in cpn_dict:
        core_set = {clean_course_code(c) for c in cpn_dict[dept].nodes() 
                    if clean_course_code(c) != "VIRTUAL_START"}
    core_set.add("CS499")
    
    # Identify general education courses
    general_courses = set(df_kesmmwad[df_kesmmwad["KesmNo"] == 11]["MadaNo"].dropna().apply(clean_course_code))
    
    # Count completed electives
    elective_courses_completed = 0
    for c in completed:
        c_clean = clean_course_code(c)
        if c_clean not in core_set and c_clean not in general_courses:
            elective_courses_completed += 1
    
    # Calculate remaining elective requirements
    elective_courses_needed = max(0, dept_elective_count - elective_courses_completed)
    elective_credits_needed = elective_courses_needed * 3
    
    print(f"📊 Already completed {elective_courses_completed} electives; "
          f"need {elective_courses_needed} more elective courses")

    # Required core courses
    required_courses = [c for c in core_set if clean_course_code(c) not in completed]
    print(f"📊 Core courses needed: {required_courses}")

    # Build prerequisite map
    prerequisites_map = defaultdict(set)
    course_order = []
    
    if dept in cpn_dict:
        G = cpn_dict[dept]
        course_order, path_gpa = find_optimal_path(
            G, completed, current_gpa, current_whdat, dept, model, model_features_list
        )
        print(f"🔍 Found path with {len(course_order)} courses using topological sort")
        
        for course in course_order:
            if course in G:
                prerequisites_map[course] = {clean_course_code(p) for p in G.predecessors(course) 
                                            if clean_course_code(p) != "VIRTUAL_START"}
    
    # Sort required courses by topological order
    required_courses = [c for c in course_order if c in required_courses]
    
    # Initialize plan
    plan = []
    cumulative_credits = current_whdat
    scheduled_courses = set(completed)
    cs499_scheduled = "CS499" in completed
    term_count = start_term - 1
    remaining_total = 134 - cumulative_credits

    # Use path optimization for small graphs (<=15 courses)
    if dept in cpn_dict and len(required_courses) <= 15:
        print("🔍 Using advanced path optimization for GPA enhancement")
        core_plan, final_gpa, cumulative_credits, elective_courses_needed = find_best_gpa_path_for_student(
            dept, cpn_dict, model, tfidf, model_features_list, credit_map, level_map,
            get_max_credits_per_term, current_gpa, cumulative_credits, completed, elective_courses_needed
        )
        
        # Add electives to each term
        cumulative = current_whdat
        new_core_plan = []
        current_gpa_temp = current_gpa
        
        for term_courses, term_gpa, term_credits in core_plan:
            # 1. Ensure minimum credits (14) for non-final terms
            remaining_total = 134 - cumulative
            min_required = min(14, remaining_total) if cumulative + term_credits < 134 else 0
            
            if term_credits < min_required:
                term_courses, term_credits, added = ensure_min_credits(
                    term_courses, term_credits, elective_courses_needed,
                    level_map, max_credits_per_term, min_required
                )
                elective_courses_needed -= added
                
                # Re-predict GPA if we added courses
                if added > 0:
                    term_gpa, term_credits = predict_term_gpa(
                        term_courses, current_gpa_temp, cumulative, dept,
                        model, tfidf, model_features_list, credit_map
                    )
            
            # 2. Add more electives if space available
            available_space = max_credits_per_term - term_credits
            while available_space >= 3 and elective_courses_needed > 0:
                candidate = term_courses + ["ELECTIVE"]
                if has_level_conflict_in_term(candidate, level_map):
                    break
                
                candidate_gpa, candidate_credits = predict_term_gpa(
                    candidate, current_gpa_temp, cumulative, dept,
                    model, tfidf, model_features_list, credit_map
                )
                
                # Only add if it doesn't harm GPA significantly
                if candidate_gpa >= term_gpa - 0.5:
                    term_courses = candidate
                    term_credits = candidate_credits
                    term_gpa = candidate_gpa
                    available_space = max_credits_per_term - term_credits
                    elective_courses_needed -= 1
                    print(f"➕ Added ELECTIVE to term")
                else:
                    break
            
            # Update for next term
            cumulative += term_credits
            current_gpa_temp = term_gpa
            max_credits_per_term = get_max_credits_per_term(current_gpa_temp)
            
            new_core_plan.append((term_courses, term_gpa, term_count + 1))
            term_count += 1
            
            print(f"✓ Scheduled term {term_count}: {term_courses} ({term_credits} credits)")
            print(f"📈 Predicted GPA: {term_gpa:.2f}")
            print(f"📊 Cumulative Credits: {cumulative}")
        
        plan = new_core_plan
        cumulative_credits = cumulative
    else:
        # Fallback to term-by-term optimization for large graphs
        print("🔍 Using term-by-term optimization")
        while required_courses or elective_courses_needed > 0:
            term_courses = []
            term_credits = 0
            term_levels = set()
            term_count += 1
            remaining_total = 134 - cumulative_credits
            
            # Calculate minimum credits for this term
            min_required = min(14, remaining_total) if remaining_total > 0 else 0
            
            # Get available courses (prerequisites met)
            available_courses = []
            for course in required_courses:
                prereqs_met = all(p in scheduled_courses for p in prerequisites_map.get(course, set()))
                if prereqs_met:
                    available_courses.append(course)
            
            # Add elective slots if needed
            if elective_courses_needed > 0:
                available_courses.extend(["ELECTIVE"] * elective_courses_needed)
            
            # =============================================================
            # ENHANCED CS499 SCHEDULING - PRIORITIZE ADDING TO EXISTING TERM
            # =============================================================
            # Check if CS499 is eligible but not scheduled
            cs499_eligible = (not cs499_scheduled and 
                              cumulative_credits >= 100 and 
                              "CS499" in available_courses and 
                              all(p in scheduled_courses for p in prerequisites_map.get("CS499", set())))
            
            # Always try to schedule CS499 first if eligible
            if cs499_eligible:
                term_courses.append("CS499")
                term_credits += credit_map["CS499"]
                available_courses.remove("CS499")
                if "CS499" in required_courses:
                    required_courses.remove("CS499")
                scheduled_courses.add("CS499")
                cs499_scheduled = True
                print(f"🎓 Scheduled CS499 in current term")
            
            # Optimize course selection for GPA
            if available_courses:
                selected, selected_credits = optimize_term_courses(
                    available_courses, min_required - term_credits, max_credits_per_term - term_credits, current_gpa,
                    cumulative_credits, dept, model, tfidf, model_features_list, credit_map, level_map
                )
                term_courses.extend(selected)
                term_credits += selected_credits
            
            # Add electives to meet minimum credit requirement
            if term_credits < min_required:
                term_courses, term_credits, added = ensure_min_credits(
                    term_courses, term_credits, elective_courses_needed,
                    level_map, max_credits_per_term, min_required
                )
                elective_courses_needed -= added
            
            # Update course tracking
            for course in term_courses:
                if course == "ELECTIVE":
                    elective_courses_needed -= 1
                elif course != "CS499":  # CS499 already handled
                    if course in required_courses:
                        required_courses.remove(course)
                    scheduled_courses.add(course)
            
            # Predict GPA impact
            if term_courses:
                new_gpa, actual_credits = predict_term_gpa(
                    term_courses, current_gpa, cumulative_credits, dept, 
                    model, tfidf, model_features_list, credit_map
                )
                
                cumulative_credits += actual_credits
                plan.append((term_courses, new_gpa, term_count))
                
                # Update for next term
                current_gpa = new_gpa
                max_credits_per_term = get_max_credits_per_term(current_gpa)
                
                print(f"✓ Scheduled term {term_count}: {term_courses} ({actual_credits} credits)")
                print(f"📈 Predicted GPA after term: {new_gpa:.2f}")
                print(f"📊 Cumulative Credits: {cumulative_credits}")
            else:
                if not required_courses and elective_courses_needed == 0:
                    break
                print("⚠️ Could not schedule any courses - checking for CS499 eligibility")
                
                # Special handling for CS499
                if (not cs499_scheduled and cumulative_credits >= 100 and 
                    all(p in scheduled_courses for p in prerequisites_map.get("CS499", set()))):
                    
                    term_courses = ["CS499"]
                    term_credits = 4
                    cs499_scheduled = True
                    if "CS499" in required_courses:
                        required_courses.remove("CS499")
                    
                    # Predict GPA impact
                    new_gpa, actual_credits = predict_term_gpa(
                        term_courses, current_gpa, cumulative_credits, dept, 
                        model, tfidf, model_features_list, credit_map
                    )
                    
                    cumulative_credits += actual_credits
                    term_count += 1
                    plan.append((term_courses, new_gpa, term_count))
                    
                    # Update GPA and credit limit
                    current_gpa = new_gpa
                    max_credits_per_term = get_max_credits_per_term(current_gpa)
                    
                    print(f"🎓 Scheduled CS499 as dedicated term")
                    print(f"✓ Scheduled term {term_count}: {term_courses} ({actual_credits} credits)")
                    print(f"📈 Predicted GPA after term: {new_gpa:.2f}")
                    print(f"📊 Cumulative Credits: {cumulative_credits}")
                else:
                    print("⚠️ No courses scheduled - breaking loop")
                    break

    # =========================================
    # ENHANCED CS499 HANDLING - ADD TO EXISTING TERM IF POSSIBLE
    # =========================================
    if not cs499_scheduled and cumulative_credits >= 100:
        print("⚠️ Graduation project not scheduled - adding to next available term")
        
        # Try to add to existing terms first
        added = False
        for i, (term_courses, term_gpa, term_number) in enumerate(plan):
            current_credits = calculate_term_credits(term_courses, credit_map)
            if current_credits + 4 <= get_max_credits_per_term(term_gpa):
                # Create candidate term with CS499
                candidate = term_courses + ["CS499"]
                if has_level_conflict_in_term(candidate, level_map):
                    continue
                    
                candidate_gpa, candidate_credits = predict_term_gpa(
                    candidate, term_gpa, cumulative_credits, dept,
                    model, tfidf, model_features_list, credit_map
                )
                
                # Only add if it doesn't harm GPA significantly
                if candidate_gpa >= term_gpa - 0.5:
                    plan[i] = (candidate, candidate_gpa, term_number)
                    cs499_scheduled = True
                    added = True
                    print(f"🎓 Added CS499 to term {term_number}")
                    break
        
        # If couldn't add to existing term, create new term
        if not added:
            print("⚠️ Could not add to existing terms - creating dedicated term")
            term = ["CS499"]
            term_count += 1
            new_gpa, actual_credits = predict_term_gpa(
                term, current_gpa, cumulative_credits, dept, 
                model, tfidf, model_features_list, credit_map
            )
            cumulative_credits += actual_credits
            plan.append((term, new_gpa, term_count))
            print(f"🎓 Added CS499 as dedicated term in term {term_count}")
        cs499_scheduled = True

    # Calculate final credits needed
    credits_needed = 134 - cumulative_credits
    if credits_needed > 0:
        print(f"⚠️ Need {credits_needed} more credits to graduate")
        num_electives = min(math.ceil(credits_needed / 3), elective_courses_needed)
        if num_electives > 0:
            term_count += 1
            term = ["ELECTIVE"] * num_electives
            term_credits = num_electives * 3
            
            # Predict GPA impact
            new_gpa, _ = predict_term_gpa(
                term, current_gpa, cumulative_credits, dept, 
                model, tfidf, model_features_list, credit_map
            )
            
            cumulative_credits += term_credits
            plan.append((term, new_gpa, term_count))
            print(f"➕ Added {num_electives} electives in term {term_count} to meet credit requirement")
    
    print(f"📊 Final projected credits: {cumulative_credits}/134")
    print(f"🎯 Final projected GPA: {current_gpa:.2f}")
    return plan, cumulative_credits

# Main recommendation engine
def recommend_gpa_enhancement_path(student_id, df_tgdidkid, df_mwad, df_kesmmwad, cpn_dict, 
                                   model, tfidf, model_features_list, chosen_dept=None):
    
    print("🚀 Starting GPA Enhancement Recommendation Engine...")
    student_data = df_tgdidkid[df_tgdidkid["KidNo"] == student_id]
    
    if student_data.empty:
        print(f"⚠️ Student {student_id} not found in records")
        return []

    # Create credit map and level map
    credit_map = {}
    level_map = {}
    for _, row in df_mwad.iterrows():
        course = clean_course_code(row["MadaNo"])
        credit = row["AddWhdat"]
        credit_map[course] = 3 if pd.isna(credit) else credit
        level_map[course] = extract_level(course)

    # Special courses
    credit_map["CS499"] = 4
    credit_map["ELECTIVE"] = 3
    level_map["CS499"] = 0
    level_map["ELECTIVE"] = 0
    
    fasel_nos = student_data["FaselNo"].drop_duplicates().sort_values(ascending=False).tolist()
    
    if len(fasel_nos) >= 2:
        current_fasel = fasel_nos[1]
    elif len(fasel_nos) == 1:
        current_fasel = fasel_nos[0]
    else:
        print(f"⚠️ Student {student_id} has no academic terms recorded")
        return []
    
    current_term_rows = student_data[student_data["FaselNo"] == current_fasel]
    if not current_term_rows.empty:
        current_term_row = current_term_rows.iloc[0]
    else:
        current_term_row = student_data.iloc[-1]
    
    current_gpa = float(current_term_row["Average"]) if pd.notna(current_term_row["Average"]) else 75.0
    zamano = int(current_term_row["Zamanno"])
    dept = int(current_term_row["KesmNo"])
    
    if zamano < 45:
        print(f"⚠️ Student {student_id} is not from current academic year (Zamano={zamano}). Only Zamano=47 students are processed.")
        return []
    
    completed, recalculated_whdat = get_completed_courses_and_credits(
        df_tgdidkid, student_id, df_mwad, max_fasel=current_fasel
    )
    current_whdat = recalculated_whdat
    
    print(f"\n📊 Student Status (as of term {current_fasel}):")
    print(f"- Department: {dept} ({'General' if dept == 11 else 'Specialized'})")
    print(f"- Completed Credits: {current_whdat} (recalculated from passed courses)")
    print(f"- Current GPA: {current_gpa:.2f}")
    print(f"- Current Term: {current_fasel}")
    print(f"- Academic Year: {zamano}")
    print(f"- Completed Courses: {', '.join(sorted(completed))}")
    
    if current_whdat >= 134:
        print("\n🎉 Congratulations! You have already completed the required 134 credits for graduation.")
        return []
    
    cs499_completed = "CS499" in completed
    if cs499_completed:
        print("\n🎓 You have already completed the Graduation Project (CS499).")
    
    if dept != 11:
        print("\n📚 Specialized Department GPA Enhancement Plan")
        plan, total_credits = recommend_specialized_path(
            dept, current_whdat, completed, df_kesmmwad, df_mwad, cpn_dict, 
            current_gpa, model, tfidf, credit_map, level_map, model_features_list,
            start_term=current_fasel + 1
        )
        cumulative = current_whdat
        for term_courses, term_gpa, term_number in plan:
            term_credits = calculate_term_credits(term_courses, credit_map)
            cumulative += term_credits
            print(f"\nTerm {term_number}:")
            print(f"- Courses: {', '.join(term_courses)}")
            print(f"- Credits: {term_credits} ({'✅ Meets minimum' if term_credits >= 14 else '⚠️ Below minimum'})")
            print(f"- Predicted GPA: {term_gpa:.2f}")
            print(f"- Cumulative Credits: {cumulative}")
            if "CS499" in term_courses and not cs499_completed:
                print("- Note: 🎓 Graduation Project (CS499) included")
                cs499_completed = True
        print(f"\n🎓 Total projected credits: {cumulative}/134")
        return [term_courses for term_courses, _, _ in plan]
    
    if current_whdat >= 45:
        print("\n🔹 You've reached 45 credits - Time to specialize!")
        if chosen_dept is None:
            # Default to department 5 if not provided
            chosen_dept = 5
            print("⚠️ No department chosen - defaulting to IN (5)")
        print(f"Selected department: {chosen_dept}")
        print("\nGenerating specialized GPA enhancement plan...")
        specialized_plan, total_credits = recommend_specialized_path(
            chosen_dept, current_whdat, completed, 
            df_kesmmwad, df_mwad, cpn_dict, current_gpa, model, tfidf, credit_map, level_map, model_features_list,
            start_term=current_fasel + 1
        )
        
        cumulative = current_whdat
        for term_courses, term_gpa, term_number in specialized_plan:
            term_credits = calculate_term_credits(term_courses, credit_map)
            cumulative += term_credits
            print(f"\nTerm {term_number}:")
            print(f"- Courses: {', '.join(term_courses)}")
            print(f"- Credits: {term_credits} ({'✅ Meets minimum' if term_credits >= 14 else '⚠️ Below minimum'})")
            print(f"- Predicted GPA: {term_gpa:.2f}")
            print(f"- Cumulative Credits: {cumulative}")
            if "CS499" in term_courses and not cs499_completed:
                print("- Note: 🎓 Graduation Project (CS499) included")
                cs499_completed = True
        print(f"\n🎓 Total projected credits: {cumulative}/134")
        return [term_courses for term_courses, _, _ in specialized_plan]
    
    print("\n🎯 General Department GPA Enhancement Plan (Until 45 Credits)")
    general_courses = set(df_kesmmwad[df_kesmmwad["KesmNo"] == 11]["MadaNo"].dropna().apply(clean_course_code))
    remaining = [c for c in general_courses if clean_course_code(c) not in completed]
    
    plan = []
    cumulative_credits = current_whdat
    term_count = 0
    max_credits_per_term = get_max_credits_per_term(current_gpa)
    
    while cumulative_credits < 45 and remaining:
        # Calculate minimum credits for this term
        remaining_total = 45 - cumulative_credits
        min_credits = min(14, remaining_total) if term_count > 0 else 0
        
        # Optimize course selection for GPA
        term_courses, term_credits = optimize_term_courses(
            remaining, min_credits, max_credits_per_term, current_gpa,
            cumulative_credits, 11, model, tfidf, model_features_list, credit_map, level_map
        )
        
        if term_courses:
            new_gpa, actual_credits = predict_term_gpa(
                term_courses, current_gpa, cumulative_credits, 11, 
                model, tfidf, model_features_list, credit_map
            )
            
            cumulative_credits += actual_credits
            term_count += 1
            term_number = current_fasel + term_count
            plan.append((term_courses, new_gpa, term_number))
            
            # Remove scheduled courses
            for course in term_courses:
                if course in remaining:
                    remaining.remove(course)
            
            current_gpa = new_gpa
            max_credits_per_term = get_max_credits_per_term(current_gpa)
            
            print(f"✓ Scheduled term {term_number}: {term_courses} ({actual_credits} credits)")
            print(f"📈 Predicted GPA after term: {new_gpa:.2f}")
        else:
            break
    
    cumulative = current_whdat
    for term_courses, term_gpa, term_number in plan:
        term_credits = calculate_term_credits(term_courses, credit_map)
        cumulative += term_credits
        print(f"\nTerm {term_number}:")
        print(f"- Courses: {', '.join(term_courses)}")
        print(f"- Credits: {term_credits} ({'✅ Meets minimum' if term_credits >= 14 else '⚠️ Below minimum'})")
        print(f"- Predicted GPA: {term_gpa:.2f}")
        print(f"- Cumulative Credits: {cumulative}")
    
    if cumulative_credits >= 45:
        print("\n🎉 You've reached 45 credits!")
        if chosen_dept is None:
            # Default to department 5 if not provided
            chosen_dept = 5
            print("⚠️ No department chosen - defaulting to IN (5)")
        print(f"Selected department: {chosen_dept}")
        print("\nGenerating specialized GPA enhancement plan...")
        new_completed = completed.union({clean_course_code(course) for term, _, _ in plan for course in term})
        specialized_plan, total_credits = recommend_specialized_path(
            chosen_dept, 
            cumulative_credits, 
            new_completed,
            df_kesmmwad,
            df_mwad,
            cpn_dict,
            plan[-1][1] if plan else current_gpa,
            model,
            tfidf,
            credit_map,
            level_map,
            model_features_list,
            start_term=current_fasel + len(plan) + 1
        )
        print("\n📚 Specialized Department GPA Enhancement Plan:")
        cumulative = cumulative_credits
        for term_courses, term_gpa, term_number in specialized_plan:
            term_credits = calculate_term_credits(term_courses, credit_map)
            cumulative += term_credits
            print(f"\nTerm {term_number}:")
            print(f"- Courses: {', '.join(term_courses)}")
            print(f"- Credits: {term_credits} ({'✅ Meets minimum' if term_credits >= 14 else '⚠️ Below minimum'})")
            print(f"- Predicted GPA: {term_gpa:.2f}")
            print(f"- Cumulative Credits: {cumulative}")
            if "CS499" in term_courses and not cs499_completed:
                print("- Note: 🎓 Graduation Project (CS499) included")
                cs499_completed = True
        print(f"\n🎓 Total projected credits: {cumulative}/134")
        return [term_courses for term_courses, _, _ in plan] + [term_courses for term_courses, _, _ in specialized_plan]
    
    return [term_courses for term_courses, _, _ in plan]