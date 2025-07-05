
import pandas as pd
from collections import defaultdict
import heapq
import re
import itertools
import time
from datetime import datetime
from sqlalchemy import create_engine

username = "root"
password = "sungjinwoo"
host = "localhost"
database = "university_db"

# Create the connection
engine = create_engine(f"mysql+pymysql://{username}:{password}@{host}/{database}")

df_mwad = pd.read_sql("SELECT * FROM mwad", engine)
df_kesmmwad = pd.read_sql("SELECT * FROM kesmmwad where zamanno = 47", engine) 
df_tgdidkid = pd.read_sql("SELECT * FROM tgdidkid", engine)
df_tanzil = pd.read_sql("SELECT * FROM tanzil", engine)


def extract_courses(course_str):
    if not course_str:
        return []
    # Split by both Arabic dash (ـ) and ASCII dash (-)
    courses = re.split(r'[ـ\-]', str(course_str))
    # Clean and filter results
    return [clean_course_code(c) for c in courses if clean_course_code(c)]


def clean_course_code(course):
    return str(course).strip().upper().replace(" ", "")

def get_completed_courses_and_credits(queryset, student_id, df_mwad, max_fasel=None, max_zamanno=None):
    # Create credit map
    credit_map = {}
    for _, row in df_mwad.iterrows():
        course = clean_course_code(row["MadaNo"])
        credit = row["AddWhdat"]
        if pd.isna(credit):
            credit = 3
        credit_map[course] = credit
    credit_map["CS499"] = 4

    # Filter by academic year and term if provided
    if max_zamanno is not None and max_fasel is not None:
        mask_prev = queryset.filter(Zamanno__lt=max_zamanno)
        mask_current = queryset.filter(Zamanno=max_zamanno, FaselNo__lte=max_fasel)
        student_rows = mask_prev | mask_current
    else:
        student_rows = queryset
    
    completed_courses = set()
    completed_credits = 0

    for record in student_rows:
        if record.Mongz:
            # Parse completed courses
            courses = []
            val = record.Mongz
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