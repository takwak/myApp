from django.shortcuts import render, redirect
from django.contrib import messages
from django.template.loader import get_template
from django.http import HttpResponse
from .models import Tgdidkid, Kesmmwad, Mwad, StudentDepartment, Student 
import pickle
import pandas as pd
import numpy as np
from .graduation_planner import recommend_fastest_path
from .getCredits import get_completed_courses_and_credits, clean_course_code, extract_courses
from django.db.models import Max, Min
from django.db.models import Q
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import os  
from django.apps import apps
import joblib
from .forms import StudentRegistrationForm
from .enhance_gpa import recommend_gpa_enhancement_path, extract_level
from Home.apps import HomeConfig
import logging
import traceback
from django.conf import settings


# Initialize logger at module level
logger = logging.getLogger(__name__)


def app_view(request):
    try:
        return render(request, 'App.html')
    except Exception as e:
        return HttpResponse(f"Template loading error: {e}")

def login_view(request):
    if request.method == 'POST':
        student_id = request.POST.get('student-id')
        password = request.POST.get('password')
        
        if Tgdidkid.objects.filter(KidNo=student_id).exists():
            request.session['student_id'] = student_id
            return redirect('Home')
        else:
            messages.error(request, 'رقم القيد غير موجود في النظام')
            return render(request, 'LogIn.html', {
                'student_id_value': student_id,
                'password_value': password
            })
    
    return render(request, 'LogIn.html')

def logout_view(request):
    if 'student_id' in request.session:
        del request.session['student_id']
    if 'current_department' in request.session:
        del request.session['current_department']
    if 'current_goal' in request.session:
        del request.session['current_goal']
    return redirect('LogIn')


def sign_view(request):
    if request.method == 'POST':
        form = StudentRegistrationForm(request.POST)
        if form.is_valid():
            student_id = form.cleaned_data['student_id']
            
            # Use the correct table name for Tgdidkid
            if not Tgdidkid.objects.filter(KidNo=student_id).exists():
                messages.error(request, 'رقم القيد غير مسجل في الكلية')
                return render(request, 'SignUp.html', {'form': form})
            
            form.save()
            messages.success(request, 'تم إنشاء الحساب بنجاح! يمكنك الآن تسجيل الدخول.')
            return redirect('LogIn')
    else:
        form = StudentRegistrationForm()
    
    return render(request, 'SignUp.html', {'form': form})

def profile_view(request):
    student_id = request.session.get('student_id')
    
    if not student_id:
        return redirect('LogIn')
    
    try:
        # Get student personal details from Student model
        student_details = Student.objects.get(student_id=student_id)
        
        # Get academic records
        student_records = Tgdidkid.objects.filter(KidNo=student_id).order_by('-FaselNo', '-Zamanno')
        
        if not student_records.exists():
            return render(request, 'profile.html', {'error': 'لا توجد سجلات أكاديمية للطالب'})
        
        student = student_records.first()
        
        # Department mapping
        department_map = {
            5: "نظم الانترنت",
            6: "علوم الحاسوب",
            7: "هندسة البرمجيات",
            8: "الشبكات والاتصالات",
            10: "الوسائط المتعددة",
            11: "القسم العام",
            14: "نظم المعلومات",
        }
        
        # Calculate completed credits
        df_mwad = pd.DataFrame.from_records(Mwad.objects.values('MadaNo', 'AddWhdat'))
        completed_courses, completed_credits = get_completed_courses_and_credits(
            student_records, 
            student_id, 
            df_mwad,
            student.FaselNo,
            student.Zamanno
        )
        
        # Calculate progress
        required_credits = 134
        progress_percent = min(100, round((completed_credits / required_credits) * 100))
        
        # Get saved plans
        saved_plans = StudentPlan.objects.filter(student_id=student_id).order_by('-created_at')
        
        # Prepare context
        context = {
            # Personal details from Student model
            'full_name': student_details.full_name,
            'student_number': student_id,
            'email': student_details.email,
            'mobile': student_details.mobile,
            'birthdate': student_details.birthdate.strftime('%d/%m/%Y'),
            
            # Academic details
            'college': 'كلية تقنية المعلومات',
            'department': department_map.get(student.KesmNo, "قسم غير معروف"),
            'academic_level': determine_academic_level(completed_credits),
            'start_date': student_details.start_date.strftime('%B %Y'),
            'expected_graduation': student_details.expected_graduation.strftime('%B %Y'),
            
            # Progress metrics
            'gpa': student.Average or 0.0,
            'completed_credits': completed_credits,
            'progress_percent': progress_percent,
            'completed_terms': student_records.count() - 1,  # Exclude current term
            'required_credits': required_credits,
            'student_id': student_id,
            'saved_plans': saved_plans,
        }
        return render(request, 'profile.html', context)
        
    except Student.DoesNotExist:
        return render(request, 'profile.html', {'error': 'الطالب غير مسجل في النظام'})
    except Exception as e:
        return render(request, 'profile.html', {'error': f'حدث خطأ: {str(e)}'})

# Add this to your views.py
def load_cpn_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pickle_path = os.path.join(base_dir, 'Home', 'cpn_dict.pkl')
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)
    return {}

# Preload the CPN data at app startup
cpn_dict = load_cpn_data()

def curriculum_map(request, department_id):
    try:
        # Skip if General department
        if int(department_id) == 11:
            return JsonResponse({'nodes': [], 'links': []})
        
        # Get the graph for the requested department
        graph = cpn_dict.get(int(department_id))
        if not graph:
            return JsonResponse({'error': 'Department not found'}, status=404)
        
        # Prepare nodes with course names and credits
        nodes = []
        for node in graph.nodes():
            # Try to get course name and credits from database
            try:
                course = Mwad.objects.get(MadaNo=node)
                name = course.MadaName
                credits = course.AddWhdat or 3  # Default to 3 if not available
            except Mwad.DoesNotExist:
                name = f"المادة {node}"
                credits = 3
            
            nodes.append({
                'id': node,
                'name': name,
                'credits': credits
            })
        
        # Prepare links
        links = []
        for edge in graph.edges():
            links.append({
                'source': edge[0],
                'target': edge[1]
            })
        
        return JsonResponse({'nodes': nodes, 'links': links})
    
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

def homepage(request):
    student_id = request.session.get('student_id')
    
    if not student_id:
        return redirect('LogIn')
    
    student_records = Tgdidkid.objects.filter(FacNo=8, KidNo=student_id).order_by('-FaselNo', '-Zamanno')
    
    if student_records.exists():
        if len(student_records) > 1:
            student = student_records[1]
        else:
            student = student_records.first()
        
        department_map = {
            5: "نظم الانترنت",
            6: "علوم الحاسوب",
            7: "هندسة البرمجيات",
            8: "الشبكات والاتصالات",
            10: "الوسائط المتعددة",
            11: "القسم العام",
            14: "نظم المعلومات",
        }
        
        dept_name = department_map.get(student.KesmNo, "قسم غير معروف")
        
        mwad_qs = Mwad.objects.all()
        df_mwad = pd.DataFrame.from_records(mwad_qs.values('MadaNo', 'AddWhdat'))
        
        if len(student_records) > 1:
            max_fasel = student.FaselNo
            max_zamanno = student.Zamanno
        else:
            max_fasel = None
            max_zamanno = None
        
        completed_courses, completed_credits = get_completed_courses_and_credits(
            student_records, 
            student_id, 
            df_mwad,
            max_fasel=max_fasel,
            max_zamanno=max_zamanno
        )
        
        total_credits = 134
        progress_percent = round((completed_credits / total_credits) * 100)
        
        current_goal = request.session.get('current_goal', '')
        current_department = request.session.get('current_department', '')
        
        # Prepare departments map for dropdown
        departments_map = {}
        for dept_id in cpn_dict.keys():
            # Only include departments that exist in our data
            if dept_id in department_map:
                departments_map[dept_id] = department_map[dept_id]
        
        context = {
            'kidno': student.KidNo,
            'college': 'كلية تقنية المعلومات',
            'department': dept_name,
            'gpa': student.Average or 0.0,
            'completed_credits': completed_credits,
            'total_credits': total_credits,
            'progress_percent': progress_percent,
            'current_goal': current_goal,
            'current_department': current_department,
            'student_id': student_id,
            'student_department_id': student.KesmNo,
            'departments_map': departments_map
        }
    else:
        context = {
            'error': 'الطالب غير موجود في الكلية',
            'progress_percent': 0,
            'student_id': student_id,
            'departments_map': {}
        }
    
    return render(request, 'Home.html', context)

def get_course_name_map():
    course_name_map = {
        "ELECTIVE": "مادة اختيارية",
        "CS499": "مشروع التخرج"
    }
    for mwad in Mwad.objects.all():
        course_name_map[clean_course_code(mwad.MadaNo)] = mwad.MadaName
    return course_name_map

def determine_academic_level(credits):
    if credits >= 100:
        return "السنة الرابعة"
    elif credits >= 67:
        return "السنة الثالثة"
    elif credits >= 34:
        return "السنة الثانية"
    return "السنة الأولى"

def process_completed_terms(tgdidkid_qs, course_name_map):
    # Get distinct terms ordered by term number (FaselNo)
    ordered = tgdidkid_qs.order_by('FaselNo').values('Zamanno', 'FaselNo').distinct()
    ordered_list = list(ordered)
    
    if not ordered_list:
        return [], 0, set(), None

    # Find the highest term number
    highest_fasel = max(term['FaselNo'] for term in ordered_list)
    
    # Process terms in order of FaselNo
    completed_terms = []
    cumulative_credits = 0
    cumulative_passed = set()

    for term in ordered_list:
        # Skip the term with the highest FaselNo (current term)
        if term['FaselNo'] == highest_fasel:
            continue
            
        records = tgdidkid_qs.filter(Zamanno=term['Zamanno'], FaselNo=term['FaselNo'])
        term_courses = []
        term_credits = 0

        for record in records:
            registered = set(extract_courses(record.TheseMwad))
            passed = set(extract_courses(record.Mongz))

            term_passed = registered & passed
            term_failed = registered - passed
            cumulative_passed |= term_passed

            for course in term_passed:
                name = course_name_map.get(course, course)
                credit = getattr(Mwad.objects.filter(MadaNo=course).first(), 'AddWhdat', 3)
                term_courses.append({'name': name, 'status': 'passed'})
                term_credits += credit

            for course in term_failed:
                name = course_name_map.get(course, course)
                term_courses.append({'name': name, 'status': 'failed'})

        cumulative_credits += term_credits
        completed_terms.append({
            'year': term['Zamanno'],
            'number': term['FaselNo'],
            'courses': term_courses,
            'term_credits': term_credits,
            'cumulative_credits': cumulative_credits
        })

    # Sort terms by FaselNo to ensure correct order
    completed_terms.sort(key=lambda x: x['number'])
    
    return completed_terms, cumulative_credits, cumulative_passed, highest_fasel

def get_last_completed_term(tgdidkid_qs):
    ordered_terms = tgdidkid_qs.order_by('-FaselNo')
    if ordered_terms.count() >= 2:
        return ordered_terms[1]  # Second most recent (last completed)
    return ordered_terms.first()



def academic_path(request):

    # Initialize context with default values
    context = {
        'completed_credits': 0,
        'gpa': 0.0,
        'academic_level': "السنة الأولى",
        'remaining_terms': 0,
        'completed_terms': [],
        'future_terms': [],
        'student_id': request.GET.get('id'),
        'goal': request.GET.get('goal'),
        'show_department_selection': False,
        'available_departments': [5, 6, 7, 8, 10, 14],
        'department_names': {
            5: "نظم الانترنت", 6: "علوم الحاسوب", 7: "هندسة البرمجيات",
            8: "الشبكات والاتصالات", 10: "الوسائط المتعددة", 14: "نظم المعلومات"
        },
        'current_department': None,
        'chosen_department': None,
        'loading': False,
        'error': None,
        'is_specialized': False,
        'message_to_user': None
    }

    if not context['student_id']:
        return render(request, 'AcademicPath.html', context)

    try:
        # Get student records
        tgdidkid_qs = Tgdidkid.objects.filter(KidNo=context['student_id'])
        if not tgdidkid_qs.exists():
            context['error'] = "لا يوجد بيانات للطالب المحدد"
            return render(request, 'AcademicPath.html', context)

        # Get last completed term info
        last_completed_term = get_last_completed_term(tgdidkid_qs)
        context['gpa'] = float(getattr(last_completed_term, 'Average', 0.0))
        context['current_department'] = getattr(last_completed_term, 'KesmNo', None)
        
        # Determine department status
        context['is_specialized'] = context['current_department'] != 11  # 11 is general department
        
        # Handle department selection
        if context['is_specialized']:
            context['chosen_department'] = context['current_department']
        else:
            try:
                student_dept = StudentDepartment.objects.filter(student_id=context['student_id']).first()
                if student_dept:
                    context['chosen_department'] = student_dept.department_id
            except Exception as e:
                context['error'] = f"خطأ في استرجاع بيانات التخصص: {str(e)}"
                return render(request, 'AcademicPath.html', context)

        context['show_department_selection'] = not context['is_specialized']

        # Prepare course data
        df_mwad = pd.DataFrame.from_records(Mwad.objects.values('MadaNo', 'AddWhdat'))
        max_fasel = last_completed_term.FaselNo
        max_zamanno = last_completed_term.Zamanno

        # Calculate completed credits and academic level
        completed_courses, completed_credits = get_completed_courses_and_credits(
            tgdidkid_qs, context['student_id'], df_mwad, max_fasel, max_zamanno
        )
        context['completed_credits'] = completed_credits
        context['academic_level'] = determine_academic_level(completed_credits)

        # Process completed terms
        course_name_map = get_course_name_map()
        completed_terms, cumulative_credits, cumulative_passed, highest_term = process_completed_terms(
            tgdidkid_qs, course_name_map
        )
        context['completed_terms'] = completed_terms

        # Generate academic path if goal is specified
        if context['goal']:
            if not context['chosen_department']:
                context['message_to_user'] = "الرجاء اختيار التخصص أولاً قبل تحديد الهدف."
            else:
                context['loading'] = True
                try:
                    # Prepare data for path generation
                    df_tgdidkid = pd.DataFrame.from_records(tgdidkid_qs.values())
                    current_academic_year = 47  # Or get dynamically if needed
                    df_kesmmwad = pd.DataFrame.from_records(
                        Kesmmwad.objects.filter(ZamanNo=current_academic_year).values('KesmNo', 'MadaNo')
                    )
                    
                    # Load prerequisite dictionary
                    cpn_path = os.path.join(settings.BASE_DIR, 'Home', 'cpn_dict.pkl')
                    if not os.path.exists(cpn_path):
                        raise FileNotFoundError(f"Prerequisite file not found at {cpn_path}")
                    
                    cpn_dict = joblib.load(cpn_path)


                    # Get models from app config
                    home_config = apps.get_app_config('Home')
                    
                    # Validate models
                    if not hasattr(home_config, 'gpa_model') or not hasattr(home_config, 'tfidf'):
                        raise RuntimeError("Prediction models not loaded in app config")
                    
                    if home_config.gpa_model is None or home_config.tfidf is None:
                        raise RuntimeError("One or more models are None")

                    # Generate path based on goal
                    if context['goal'] == 'fast-grad':
                        plan = recommend_fastest_path(
                            student_id=context['student_id'],
                            df_tgdidkid=df_tgdidkid,
                            df_mwad=df_mwad,
                            df_kesmmwad=df_kesmmwad,
                            cpn_dict=cpn_dict,
                            target_dept=context['chosen_department']
                        )
                    elif context['goal'] == 'gpa-goal':
                        plan = recommend_gpa_enhancement_path(
                            student_id=context['student_id'],
                            df_tgdidkid=df_tgdidkid,
                            df_mwad=df_mwad,
                            df_kesmmwad=df_kesmmwad,
                            cpn_dict=cpn_dict,
                            model=home_config.gpa_model,
                            tfidf=home_config.tfidf,
                            model_features_list=home_config.model_features,
                            chosen_dept=context['chosen_department']
                        )

                    # Process generated plan
                    future_terms = []
                    cumulative = completed_credits
                    for i, term_courses in enumerate(plan):
                        term = []
                        term_credits = 0
                        for course in term_courses:
                            clean_code = clean_course_code(course)
                            name = course_name_map.get(clean_code, clean_code)
                            mwad_obj = Mwad.objects.filter(MadaNo=clean_code).first()
                            credit = getattr(mwad_obj, 'AddWhdat', 3) if mwad_obj else 3
                            if clean_code == "CS499":
                                credit = 4
                            term.append({'name': name, 'status': 'planned'})
                            term_credits += credit

                        cumulative += term_credits
                        future_terms.append({
                            'number': highest_term + i,
                            'courses': term,
                            'term_credits': term_credits,
                            'cumulative_credits': cumulative
                        })

                    context['future_terms'] = future_terms
                    context['remaining_terms'] = len(future_terms)
                    
                except Exception as e:
                    logger.error(f"Error generating academic path: {str(e)}\n{traceback.format_exc()}")
                    context['error'] = f"حدث خطأ أثناء إنشاء المسار: {str(e)}"
                finally:
                    context['loading'] = False
        
        # Set appropriate messages
        if not context['goal'] and context['chosen_department']:
            context['message_to_user'] = "الرجاء اختيار هدف دراسي لرؤية المسار الأكاديمي"
        elif not context['is_specialized'] and not context['chosen_department']:
            context['message_to_user'] = "الرجاء اختيار التخصص أولاً لرؤية المسار الأكاديمي"

    except Exception as e:
        logger.error(f"Unexpected error in academic_path: {str(e)}\n{traceback.format_exc()}")
        context['error'] = f"حدث خطأ غير متوقع: {str(e)}"

    return render(request, 'AcademicPath.html', context)

    
@csrf_exempt
def save_department(request):
    if request.method == 'POST':
        department = request.POST.get('department')
        student_id = request.POST.get('student_id')
        
        if not student_id or not department:
            return JsonResponse({'status': 'error', 'message': 'Missing parameters'}, status=400)
        
        try:
            # Create or update department choice
            StudentDepartment.objects.update_or_create(
                student_id=student_id,
                defaults={'department_id': int(department)}
            )
            return JsonResponse({'status': 'success'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    
    return JsonResponse({'status': 'error', 'message': 'Invalid method'}, status=405)


@csrf_exempt
def save_goal(request):
    if request.method == 'POST':
        goal = request.POST.get('goal', '')
        request.session['current_goal'] = goal
        return JsonResponse({'status': 'success'})
    return JsonResponse({'status': 'error'}, status=400)


    # views.py (add these views)
@csrf_exempt
def save_plan(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            student_id = request.session.get('student_id')
            
            # Get current plan data (this would come from the session or calculation)
            current_department = request.session.get('current_department', '')
            current_goal = request.session.get('current_goal', '')
            
            # Create new plan
            plan = StudentPlan.objects.create(
                student_id=student_id,
                plan_name=data.get('name'),
                department_id=current_department,
                department_name=department_map.get(int(current_department), "غير محدد"),
                goal=current_goal,
                courses=[],  # This would be populated with actual courses
                duration=0  # This would be calculated
            )
            
            return JsonResponse({'success': True, 'plan_id': plan.id})
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    return JsonResponse({'success': False, 'error': 'Invalid method'})

@csrf_exempt
def delete_plan(request):
    if request.method == 'DELETE':
        try:
            plan_id = request.GET.get('id')
            student_id = request.session.get('student_id')
            
            plan = StudentPlan.objects.get(id=plan_id, student_id=student_id)
            plan.delete()
            
            return JsonResponse({'success': True})
        except StudentPlan.DoesNotExist:
            return JsonResponse({'success': False, 'error': 'الخطة غير موجودة'})
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    return JsonResponse({'success': False, 'error': 'Invalid method'})

