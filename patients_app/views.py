from django.shortcuts import render
from rest_framework import generics
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from .models import Patient
from .serializers import (
    PatientSerializer,
    PatientVideoSerializer,
    PatientDoctorIDSerializer,
)


class PatientCreateAPIView(generics.CreateAPIView):
    queryset = Patient.objects.all()
    serializer_class = PatientSerializer

    def create_patient(self, request):
        serializer = PatientSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


patient_create_view = PatientCreateAPIView.as_view()


class PatientListAPIView(generics.ListAPIView):
    queryset = Patient.objects.all()
    serializer_class = PatientSerializer


patient_list_view = PatientListAPIView.as_view()


class PatientUpdateVideoAPIView(generics.UpdateAPIView):
    queryset = Patient.objects.all()
    serializer_class = PatientVideoSerializer
    lookup_field = "pk"

    def partial_update(self, request, *args, **kwargs):
        instance = self.get_object()
        patient_video_file = request.FILES.get("patient_video")
        if patient_video_file:
            instance.patient_video = patient_video_file
            instance.save()
            return Response(
                {"message": "Patient video updated successfully."},
                status=status.HTTP_200_OK,
            )
        else:
            return Response(
                {"error": "No patient video file provided."},
                status=status.HTTP_400_BAD_REQUEST,
            )


patient_update_video_view = PatientUpdateVideoAPIView.as_view()


class PatientUpdateDoctorIDAPIView(generics.UpdateAPIView):
    queryset = Patient.objects.all()
    serializer_class = PatientDoctorIDSerializer
    lookup_field = "pk"

    def partial_update(self, request, *args, **kwargs):
        instance = self.get_object()
        patient_doctor_id = request.get("doctor_id")
        if patient_doctor_id:
            instance.patient_doctor_id = patient_doctor_id
            instance.save()
            return Response(
                {"message": "Doctor ID updated successfully."},
                status=status.HTTP_200_OK,
            )
        else:
            return Response(
                {"error": "No doctor ID provided."},
                status=status.HTTP_400_BAD_REQUEST,
            )


patient_update_doctor_ID_view = PatientUpdateDoctorIDAPIView.as_view()


# class PatientUpdateAPIView(generics.UpdateAPIView):
#     queryset = Patient.objects.all()
#     serializer_class = PatientSerializer
#     lookup_field = "id"

#     def perform_update(self, serializer):
#         instance = serializer.save()
#         patient_video_file = self.request.FILES.update("patient_video")
#         if patient_video_file:
#             instance.patient_video = patient_video_file
#             instance.save()


# patient_update_video_view = PatientUpdateAPIView.as_view()


class PatientDeleteAPIView(generics.DestroyAPIView):
    def delete_patient():
        pass
