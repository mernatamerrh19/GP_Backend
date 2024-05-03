from django.contrib import admin

from doctors_app.models import Doctor


# Register your models here.
@admin.register(Doctor)
class DoctorAdmin(admin.ModelAdmin):
    pass