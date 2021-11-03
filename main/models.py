from django.db import models

# Create your models here.

class Images_and_Prediction(models.Model):
    id = models.IntegerField(primary_key=True, unique=True)
    image = models.BinaryField()
    detected_numbers = models.BinaryField()
    prediction = models.CharField(max_length=200)

    def __str__(self):
        return str(str(self.id) + ", prediction: " + str(self.prediction))