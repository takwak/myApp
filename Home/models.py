from django.db import models
from django.contrib.auth.models import AbstractUser

class Tgdidkid(models.Model):
    FacNo = models.IntegerField()
    KidNo = models.CharField(max_length=11)
    FaselNo = models.SmallIntegerField()
    Zamanno = models.IntegerField()
    KesmNo = models.IntegerField()
    Door = models.SmallIntegerField(null=True, blank=True)
    Total = models.FloatField(null=True, blank=True)
    TotalNekat = models.FloatField(null=True, blank=True)
    TotalWhdat = models.SmallIntegerField(null=True, blank=True)
    Nesba = models.FloatField(null=True, blank=True)
    Average = models.FloatField(null=True, blank=True)
    IsalNo = models.CharField(max_length=11, null=True, blank=True)
    IsalDate = models.DateField(null=True, blank=True)
    Mahgoba = models.SmallIntegerField(null=True, blank=True)
    TheseMwad = models.TextField(null=True, blank=True)
    Mongz = models.TextField(null=True, blank=True)
    NotMongz = models.TextField(null=True, blank=True)
    WhdatMosgla = models.SmallIntegerField(null=True, blank=True)
    WhdatMoaada = models.SmallIntegerField(null=True, blank=True)
    WhdatMongz = models.SmallIntegerField(null=True, blank=True)
    Lock1 = models.SmallIntegerField(null=True, blank=True)
    EntharType = models.SmallIntegerField(null=True, blank=True)
    AddEntharat = models.SmallIntegerField(null=True, blank=True)
    FosolEntharat = models.TextField(null=True, blank=True)
    RepAverage = models.FloatField(null=True, blank=True)
    RepNotMongz = models.TextField(null=True, blank=True)
    RepMongz = models.TextField(null=True, blank=True)
    RepWhdatMongz = models.SmallIntegerField(null=True, blank=True)
    EikafKid = models.SmallIntegerField(null=True, blank=True)
    AutoId = models.AutoField(primary_key=True)
    TasgilType = models.SmallIntegerField(null=True, blank=True, default=0)
    GetDeploma = models.SmallIntegerField(null=True, blank=True, default=0)

    class Meta:
        db_table = 'tgdidkid'

class Kesmmwad(models.Model):
    KesmNo = models.IntegerField()
    ZamanNo = models.IntegerField()
    MadaNo = models.CharField(max_length=11)
    MadaType = models.SmallIntegerField(null=True, blank=True)
    AutoId = models.AutoField(primary_key=True)

    class Meta:
        db_table = 'kesmmwad'

class Mwad(models.Model):
    FacNo = models.IntegerField()
    Level = models.SmallIntegerField()
    MadaNo = models.CharField(max_length=11)
    MadaName = models.CharField(max_length=50, null=True, blank=True)
    AddWhdat = models.SmallIntegerField(null=True, blank=True)
    MadaType = models.SmallIntegerField(null=True, blank=True)
    Asbkia = models.CharField(max_length=11, null=True, blank=True)
    Asbkia1 = models.CharField(max_length=11, null=True, blank=True)
    Asbkia2 = models.CharField(max_length=11, null=True, blank=True)
    NgahDrga = models.FloatField(null=True, blank=True)
    TeacherNo = models.IntegerField(null=True, blank=True)
    MType = models.CharField(max_length=10, null=True, blank=True)
    TeachORNo = models.SmallIntegerField(null=True, blank=True)
    TanzilShart = models.SmallIntegerField(null=True, blank=True)
    BarCode = models.IntegerField(null=True, blank=True)
    AutoId = models.AutoField(primary_key=True)
    MadaNameE = models.CharField(max_length=50, null=True, blank=True)
    MadaNoE = models.CharField(max_length=11, null=True, blank=True)

    class Meta:
        db_table = 'mwad'

class StudentDepartment(models.Model):
    student_id = models.CharField(max_length=20, primary_key=True)
    department_id = models.IntegerField()
    
    class Meta:
        db_table = 'StudentDepartment'



class Student(AbstractUser):
    student_id = models.CharField(max_length=20, unique=True, verbose_name="رقم القيد")
    email = models.EmailField(unique=True, verbose_name="البريد الإلكتروني")

    # Remove unused fields from AbstractUser
    first_name = None
    last_name = None
    username = None

    USERNAME_FIELD = 'student_id'
    REQUIRED_FIELDS = ['email']

    class Meta:
        db_table = 'home_student'  # Explicitly set table name
        verbose_name = "طالب"
        verbose_name_plural = "الطلاب"

    def __str__(self):
        return self.student_id


# models.py (add this model)
class StudentPlan(models.Model):
    student = models.ForeignKey(Tgdidkid, on_delete=models.CASCADE)
    plan_name = models.CharField(max_length=255)
    department_id = models.IntegerField()
    department_name = models.CharField(max_length=255)
    goal = models.CharField(max_length=50)
    courses = models.JSONField()  # Stores the plan courses
    duration = models.IntegerField()  # Number of terms
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.student.KidNo} - {self.plan_name}"