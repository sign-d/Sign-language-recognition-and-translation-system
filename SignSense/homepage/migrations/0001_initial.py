# Generated by Django 5.0.7 on 2024-08-03 18:59

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="Sign",
            fields=[
                (
                    "last_login",
                    models.DateTimeField(
                        blank=True, null=True, verbose_name="last login"
                    ),
                ),
                ("id", models.AutoField(primary_key=True, serialize=False)),
                ("email", models.EmailField(max_length=254, unique=True)),
                ("password", models.CharField(max_length=128)),
            ],
            options={
                "abstract": False,
            },
        ),
    ]
