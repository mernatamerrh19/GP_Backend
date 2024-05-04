from rest_framework import serializers
from .models import Doctor
from patients_app.serializers import PatientSerializer
from django.contrib.auth import authenticate
from django.utils.translation import gettext_lazy as _
from django.utils.translation import gettext as _


from django.core.exceptions import ValidationError
from django.core.validators import validate_email as django_validate_email


class DoctorSerializer(serializers.ModelSerializer):
    id = serializers.SerializerMethodField()
    email = serializers.EmailField()
    # patients = PatientSerializer(many=True, read_only=True)
    password = serializers.CharField(
        style={"input_type": "password"}, trim_whitespace=False, write_only=True
    )

    class Meta:
        model = Doctor
        fields = [
            "id",
            "first_name",
            "last_name",
            "email",
            "password",
        ]

    def get_id(self, obj):
        return obj.id

    def validate(self, attrs):
        email = attrs.get("email", None)
        if Doctor.objects.filter(email=email.lower()).exists():
            raise serializers.ValidationError(
                {"email": _("this email is already exist")}
            )
        return attrs

    def create(self, validated_data):
        validated_data["type"] = "doctor"
        user = Doctor.objects.create_user(**validated_data)

        return user


class PatientSerializer(serializers.ModelSerializer):
    email = serializers.EmailField()
    password = serializers.CharField(
        style={"input_type": "password"}, trim_whitespace=False, write_only=True
    )
    age = serializers.SerializerMethodField()

    class Meta:
        model = Doctor
        fields = [
            "id",
            "first_name",
            "last_name",
            "email",
            "password",
            "doctor",
            "age",
            "birthday",
        ]

    def get_age(self, obj):
        return obj.calculate_age

    def validate(self, attrs):
        email = attrs.get("email", None)
        if Doctor.objects.filter(email=email.lower()).exists():
            raise serializers.ValidationError(
                {"email": _("this email is already exist")}
            )
        return attrs

    def create(self, validated_data):
        validated_data["type"] = "patient"
        validated_data["is_verified"] = False
        user = Doctor.objects.create_user(**validated_data)

        return user


class AuthCustomTokenSerializer(serializers.Serializer):
    email_or_username = serializers.CharField()
    password = serializers.CharField(
        label=_(
            "Password",
        ),
        style={"input_type": "password"},
        trim_whitespace=False,
    )

    def validate(self, attrs):
        email_or_username = attrs.get("email_or_username")
        password = attrs.get("password")

        if email_or_username and password:
            user = authenticate(
                email=email_or_username,
                password=password,
            )
            if user:
                if not user.is_active:
                    raise serializers.ValidationError(_("User account is disabled."))
                elif not user.is_verified:
                    raise serializers.ValidationError(
                        _(
                            "User account is not validated yet. Please wait for doctor verification."
                        )
                    )
            else:
                raise serializers.ValidationError(
                    _("Unable to log in with provided credentials.")
                )
        else:
            raise serializers.ValidationError(
                _('Must include "email or username" and "password"')
            )

        attrs["user"] = user
        return attrs
