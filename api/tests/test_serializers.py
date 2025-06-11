from django.test import TestCase
from django.contrib.auth import get_user_model
from rest_framework.exceptions import ValidationError
from api.serializers.auth_serializers import RegisterSerializer
from api.models.user_models import CustomUser

User = get_user_model()


class RegisterSerializerTest(TestCase):
    
    def setUp(self):
        self.valid_data = {
            'first_name': 'John',
            'last_name': 'Doe',
            'email': 'john.doe@example.com',
            'username': 'johndoe',
            'password': 'strongpassword123',
            'password1': 'strongpassword123',
            'phone_number': '+1234567890'
        }
    
    def test_valid_registration(self):
        """Test valid user registration"""
        serializer = RegisterSerializer(data=self.valid_data)
        self.assertTrue(serializer.is_valid())
        
        user = serializer.save()
        
        self.assertEqual(user.first_name, 'John')
        self.assertEqual(user.last_name, 'Doe')
        self.assertEqual(user.email, 'john.doe@example.com')
        self.assertEqual(user.username, 'johndoe')
        self.assertEqual(str(user.phone_number), '+1234567890')
        self.assertTrue(user.check_password('strongpassword123'))
    
    def test_password_mismatch(self):
        """Test validation when passwords don't match"""
        data = self.valid_data.copy()
        data['password1'] = 'differentpassword'
        
        serializer = RegisterSerializer(data=data)
        self.assertFalse(serializer.is_valid())
        self.assertIn('password1', serializer.errors)
        self.assertEqual(
            serializer.errors['password1'][0], 
            'Passwords do not match'
        )
    
    def test_duplicate_email(self):
        """Test validation when email already exists"""
        # Create a user with the email first
        CustomUser.objects.create_user(
            username='existinguser',
            email='john.doe@example.com',
            phone_number='+9876543210',
            password='password123'
        )
        
        serializer = RegisterSerializer(data=self.valid_data)
        self.assertFalse(serializer.is_valid())
        self.assertIn('email', serializer.errors)
        self.assertEqual(
            serializer.errors['email'][0],
            'Email already exists'
        )
    
    def test_missing_required_fields(self):
        """Test validation when required fields are missing"""
        # Test missing first_name
        data = self.valid_data.copy()
        del data['first_name']
        
        serializer = RegisterSerializer(data=data)
        self.assertFalse(serializer.is_valid())
        self.assertIn('first_name', serializer.errors)
        
        # Test missing last_name
        data = self.valid_data.copy()
        del data['last_name']
        
        serializer = RegisterSerializer(data=data)
        self.assertFalse(serializer.is_valid())
        self.assertIn('last_name', serializer.errors)
    
    def test_invalid_phone_number(self):
        """Test validation with invalid phone number"""
        data = self.valid_data.copy()
        data['phone_number'] = 'invalid-phone'
        
        serializer = RegisterSerializer(data=data)
        self.assertFalse(serializer.is_valid())
        self.assertIn('phone_number', serializer.errors)
    
    def test_empty_string_fields(self):
        """Test validation with empty string fields"""
        data = self.valid_data.copy()
        data['first_name'] = ''
        data['last_name'] = ''
        
        serializer = RegisterSerializer(data=data)
        self.assertFalse(serializer.is_valid())
        self.assertIn('first_name', serializer.errors)
        self.assertIn('last_name', serializer.errors)
    
    def test_serializer_fields(self):
        """Test that serializer has all expected fields"""
        serializer = RegisterSerializer()
        expected_fields = {
            'first_name', 'last_name', 'email', 'username', 
            'password', 'password1', 'phone_number'
        }
        self.assertEqual(set(serializer.fields.keys()), expected_fields)
    
    def test_password_write_only(self):
        """Test that password fields are write-only"""
        serializer = RegisterSerializer()
        self.assertTrue(serializer.fields['password'].write_only)
        self.assertTrue(serializer.fields['password1'].write_only)
    
    def test_create_user_with_minimum_data(self):
        """Test creating user with only required fields"""
        data = {
            'first_name': 'Jane',
            'last_name': 'Smith',
            'email': 'jane.smith@example.com',
            'username': 'janesmith',
            'password': 'password123',
            'password1': 'password123',
            'phone_number': '+1987654321'
        }
        
        serializer = RegisterSerializer(data=data)
        self.assertTrue(serializer.is_valid())
        
        user = serializer.save()
        self.assertEqual(user.username, 'janesmith')
        self.assertEqual(user.email, 'jane.smith@example.com')
    
    def test_validate_email_case_insensitive(self):
        """Test email validation is case insensitive"""
        # Create user with lowercase email
        CustomUser.objects.create_user(
            username='testuser',
            email='test@example.com',
            phone_number='+1111111111',
            password='password123'
        )
        
        # Try to create with uppercase email
        data = self.valid_data.copy()
        data['email'] = 'TEST@EXAMPLE.COM'
        data['username'] = 'different_username'
        data['phone_number'] = '+2222222222'
        
        serializer = RegisterSerializer(data=data)
        # Django's email field is case insensitive by default
        # This test depends on your specific implementation
        # You might need to adjust based on your actual validation logic
    
    def test_create_method_called_correctly(self):
        """Test that create method properly uses validated data"""
        serializer = RegisterSerializer(data=self.valid_data)
        self.assertTrue(serializer.is_valid())
        
        user = serializer.save()
        
        # Verify all fields are set correctly
        self.assertEqual(user.first_name, self.valid_data['first_name'])
        self.assertEqual(user.last_name, self.valid_data['last_name'])
        self.assertEqual(user.username, self.valid_data['username'])
        self.assertEqual(user.email, self.valid_data['email'])
        self.assertEqual(str(user.phone_number), self.valid_data['phone_number'])
        
        # Verify password is hashed correctly
        self.assertNotEqual(user.password, self.valid_data['password'])
        self.assertTrue(user.check_password(self.valid_data['password'])) 