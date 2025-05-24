from django.db import models
from django.contrib.auth.models import AbstractUser
from django.utils.translation import gettext_lazy as _
from django.core.validators import RegexValidator
from phonenumber_field.modelfields import PhoneNumberField


class CustomUser(AbstractUser):
    
    
    phone_number = PhoneNumberField(
        unique=True,
        
        verbose_name=_('Phone Number')
    )
    
    class Meta:
        verbose_name = _('User')
        verbose_name_plural = _('Users')
        
    def __str__(self):
        return self.username