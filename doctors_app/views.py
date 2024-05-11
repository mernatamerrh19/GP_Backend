from rest_framework import generics
from rest_framework import status
from rest_framework.response import Response

from py_client.audio_file_and_transcript import process_video
from .models import Doctor
from .serializers import DoctorSerializer, AuthCustomTokenSerializer, PatientSerializer, VideoSerializer
from rest_framework import viewsets
from rest_framework.authtoken.models import Token
from rest_framework.authtoken.models import Token
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.parsers import MultiPartParser, FormParser
import subprocess


class SignUp(viewsets.ModelViewSet):
    """
    A viewset for signing up new users.
    """

    authentication_classes = []  # Disable authentication for this view
    permission_classes = []
    serializer_class = DoctorSerializer
    queryset = Doctor.objects.all()
    http_method_names = ["post"]

    def get_serializer_class(self):
        if "/doctors/patient/" == self.request.path:
            from doctors_app.serializers import PatientSerializer

            return PatientSerializer
        return self.serializer_class


class Login(generics.GenericAPIView):
    # authentication_classes = []
    permission_classes = []
    serializer_class = AuthCustomTokenSerializer

    def post(self, request, *args, **kwargs):
        serializer = self.serializer_class(
            data=request.data, context={"request": request}
        )
        serializer.is_valid(raise_exception=True)
        user = serializer.validated_data["user"]
        if request.path == "/doctors/patient-login/" and user.type == "patient":
            pass
        elif request.path == "/doctors/doctor-login/" and user.type == "doctor":
            pass
        else:
            return Response("invalid credentials")

        token, created = Token.objects.get_or_create(user=user)
        return Response(
            {"token": token.key, "id": user.pk, "email": user.email, "type": user.type}
        )


class Logout(generics.GenericAPIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, *args):
        token = Token.objects.get(user=request.user)
        token.delete()
        return Response(
            {"success": True, "detail": "Logged out!"}, status=status.HTTP_200_OK
        )


class PendingPatientRequestsView(generics.ListAPIView):
    serializer_class = PatientSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        doctor = self.request.user
        # Filter patients who are not yet verified
        return Doctor.objects.filter(doctor=doctor, is_verified=False)


class PatientVerificationView(generics.GenericAPIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, patient_id):
        try:
            doctor = request.user
            patient = Doctor.objects.get(
                id=patient_id, doctor=doctor, is_verified=False
            )
            action = request.data.get("action")  # Get the action from request data

            # Perform action based on doctor's choice
            if action == "accept":
                patient.is_verified = True
                patient.save()
                return Response(
                    {"detail": "Patient verified successfully"},
                    status=status.HTTP_200_OK,
                )
            elif action == "ignore":
                # Perform any other action here, like notifying the patient or logging the decision
                patient.delete()
                return Response(
                    {"detail": "Patient ignored. Patient data deleted."},
                    status=status.HTTP_200_OK,
                )
            else:
                return Response(
                    {"detail": "Invalid action"}, status=status.HTTP_400_BAD_REQUEST
                )
        except Doctor.DoesNotExist:
            return Response(
                {"detail": "Patient not found"}, status=status.HTTP_404_NOT_FOUND
            )


class DoctorPatientView(generics.GenericAPIView):
    serializer_class = PatientSerializer
    permission_classes = [IsAuthenticated]

    def get(self, request):
        # Get the doctor associated with the token
        try:
            token = request.headers.get("Authorization").split(" ")[1]
            user = Token.objects.get(key=token).user
            doctor = Doctor.objects.get(email=user.email)
        except Token.DoesNotExist:
            return Response(
                {"detail": "Token not found"}, status=status.HTTP_401_UNAUTHORIZED
            )
        except Doctor.DoesNotExist:
            return Response(
                {"detail": "Doctor not found"}, status=status.HTTP_404_NOT_FOUND
            )

        # Retrieve patients associated with the doctor along with their videos
        patients = doctor.patients.filter(is_verified=True)
        serializer = self.get_serializer(patients, many=True)
        return Response(serializer.data)


class VideoViewSet(generics.GenericAPIView):
    parser_classes = [MultiPartParser, FormParser]
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = VideoSerializer(data=request.data)
        if serializer.is_valid():
            # Assign the current patient (authenticated user) to the doctor field of the video
            serializer.validated_data["patient"] = request.user
            instance = serializer.save()
            # Add your video processing logic here
            response_data = {
                "video_id": instance.id,
                "video_file": instance.video_file.url,
                "uploaded_at": instance.formatted_uploaded_at(),
            }
            return Response(response_data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class DoctorCreateAPIView(generics.CreateAPIView):
    queryset = Doctor.objects.all()
    serializer_class = DoctorSerializer

    def create_doctor(self, request):
        serializer = DoctorSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


doctor_create_view = DoctorCreateAPIView.as_view()


class DoctorListAPIView(generics.ListAPIView):
    permission_classes = [IsAuthenticated]
    queryset = Doctor.objects.all()
    serializer_class = DoctorSerializer


doctor_list_view = DoctorListAPIView.as_view()
