from django.urls import path
from . import views

urlpatterns = [
    path("", views.patient_create_view),
    path("list/", views.patient_list_view),
    path("<int:pk>/update/video", views.patient_update_video_view),
    path("<int:pk>/update/doctor", views.patient_update_doctor_ID_view),
]
