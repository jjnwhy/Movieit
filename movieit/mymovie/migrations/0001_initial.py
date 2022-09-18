# Generated by Django 4.0.6 on 2022-09-06 18:50

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='NoticeTab',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=20)),
                ('passwd', models.CharField(max_length=20)),
                ('mail', models.CharField(max_length=30)),
                ('title', models.CharField(max_length=100)),
                ('cont', models.TextField()),
                ('nip', models.GenericIPAddressField()),
                ('ndate', models.DateTimeField()),
                ('readcnt', models.IntegerField()),
                ('likecnt', models.IntegerField()),
            ],
        ),
    ]
