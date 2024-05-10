from django.urls import path, include
from . import views
from rest_framework.routers import DefaultRouter

router = DefaultRouter()
router.register("doctors", views.SignUp, basename="doctors")
router.register("patient", views.SignUp, basename="patient")


urlpatterns = [
    path("", include(router.urls)),
    path("doctor-login/", views.Login.as_view(), name="Login"),
    path("patient-login/", views.Login.as_view(), name="Login"),
    path("logout/", views.Logout.as_view(), name="Logout"),
    path("", views.doctor_create_view),
    path("list/", views.doctor_list_view),
    path("all-patients/", views.DoctorPatientView.as_view()),
    path("doctor/pending-patients", views.PendingPatientRequestsView.as_view()),
    path(
        "doctor/patient-verification/<str:patient_id>",
        views.PatientVerificationView.as_view(),
    ),
    path("patient/patient-video", views.VideoViewSet.as_view(), name="Video"),
]
