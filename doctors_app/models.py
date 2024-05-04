from datetime import datetime
import random
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager, PermissionsMixin
from django.db import models


class UserManager(BaseUserManager):

    def create_user(
            self, email, first_name=None,
            last_name=None,
            password=None, is_verified=True, **extra_fields):

        if email is None:
            raise TypeError('Users should have a Email')

        user = self.model(first_name=first_name, last_name=(last_name or None),
                          email=self.normalize_email(email), **extra_fields)
        user.set_password(password)
        user.is_verified = is_verified
        user.save()
        return user

    def create_superuser(self, email, first_name=None, last_name=None, password=None, **extra_fields):
        if password is None:
            raise TypeError('Password should not be none')

        user = self.create_user(email, first_name, last_name, password, **extra_fields)
        user.is_superuser = True
        user.is_active = True
        user.is_staff = True
        user.is_verified = True
        user.save()
        return user


class Doctor(AbstractBaseUser, PermissionsMixin):
    choices = [
        ("doctor", "doctor"),
        ("patient", "patient"),
    ]
    id = models.CharField(primary_key=True, max_length=4, editable=False)
    first_name = models.CharField(max_length=120, blank=False, null=False)
    last_name = models.CharField(max_length=120, blank=False, null=False)
    email = models.EmailField(unique=True)
    type = models.CharField(default="doctor", choices=choices, max_length=50)
    birthday = models.DateField(null=True)
    doctor = models.ForeignKey(
        "self", on_delete=models.CASCADE, related_name="patients",
        blank=True, null=True,
        limit_choices_to={"type": "doctor"},
    )
    is_staff = models.BooleanField(default=False)
    is_verified = models.BooleanField(default=True)
    is_active = models.BooleanField(default=True)
    username = models.CharField(max_length=150, unique=True, blank=True)
    USERNAME_FIELD = "email"
    REQUIRED_FIELDS = ["first_name", "last_name"]

    # Specify unique related names for groups and user_permissions fields
    groups = models.ManyToManyField(
        "auth.Group", related_name="doctor_groups", blank=True
    )
    user_permissions = models.ManyToManyField(
        "auth.Permission", related_name="doctor_permissions", blank=True
    )
    objects = UserManager()

    class Meta:
        permissions = (("can_view_doctor", "Can view doctor"),)
        verbose_name = "Doctor"
        verbose_name_plural = "Doctors"

    def save(self, *args, **kwargs):
        if not self.id:
            self.id = "".join(random.choices("0123456789", k=4))
            self.username = self.email  # Set username to email
        super().save(*args, **kwargs)
        
    @property
    def calculate_age(self):
        today = datetime.today()
        age = (
            today.year
            - self.birthday.year
            - ((today.month, today.day) < (self.birthday.month, self.birthday.day))
        )
        return age
