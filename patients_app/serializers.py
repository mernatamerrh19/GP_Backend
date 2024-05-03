from rest_framework import serializers
from .models import Patient
from doctors_app.models import Doctor


class PatientSerializer(serializers.ModelSerializer):
    age = serializers.SerializerMethodField()
    patient_video = serializers.FileField()
    doctor = serializers.PrimaryKeyRelatedField(queryset=Doctor.objects.all())

    class Meta:
        model = Patient
        fields = [
            "id",
            "first_name",
            "last_name",
            "birthday",
            "age",
            "patient_video",
            "doctor",
        ]

    def get_age(self, obj):
        return obj.calculate_age


class PatientVideoSerializer(serializers.ModelSerializer):
    patient_video = serializers.FileField()

    class Meta:
        model = Patient
        fields = ["patient_video"]


class PatientDoctorIDSerializer(serializers.ModelSerializer):
    doctor_id = serializers.IntegerField()

    class Meta:
        model = Patient
        fields = ["doctor_id"]
