from django.contrib.auth.password_validation import validate_password
from rest_framework import serializers
from api.models.user_models import CustomUser
from django.core.validators import RegexValidator
class RegisterSerializer(serializers.ModelSerializer):
    first_name = serializers.CharField(required=True)
    last_name = serializers.CharField(required=True)
    password = serializers.CharField(write_only=True)
    password1 = serializers.CharField(write_only=True)
    
    class Meta:
        model = CustomUser
        fields = ('first_name', 'last_name', 'email', 'username', 'password','password1', 'phone_number')

    def create(self, validated_data):
        user = CustomUser.objects.create_user(
            first_name=validated_data['first_name'],
            last_name=validated_data['last_name'],
            username=validated_data['username'],
            email=validated_data['email'],
            password=validated_data['password'],
            phone_number=validated_data['phone_number']
        )
        return user
    
    def validate_email(self, value):
        if CustomUser.objects.filter(email=value).exists():
            raise serializers.ValidationError("Email already exists")
        return value

    def validate_password1(self, value):
        if value != self.initial_data['password']:
            raise serializers.ValidationError("Passwords do not match")
        return value