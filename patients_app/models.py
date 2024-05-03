from django.db import models
from datetime import datetime
from doctors_app.models import Doctor


class Patient(models.Model):
    first_name = models.CharField(max_length=120)
    last_name = models.CharField(max_length=120)
    birthday = models.DateField()
    patient_video = models.FileField(upload_to="video/%Y/%m/%d", null=True)
    doctor = models.ForeignKey(
        Doctor, on_delete=models.CASCADE, null=True
    )

    @property
    def calculate_age(self):
        today = datetime.today()
        age = (
            today.year
            - self.birthday.year
            - ((today.month, today.day) < (self.birthday.month, self.birthday.day))
        )
        return age
