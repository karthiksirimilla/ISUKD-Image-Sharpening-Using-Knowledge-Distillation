import os
import torch
from models.teacher_model import ImageRestorationTeacher
from models.student_model import VideoSharpeningStudent

def load_models(device='cpu'):
    # Get absolute path to the models directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.abspath(os.path.join(base_dir, '../models'))
    student_path = os.path.join(models_dir, 'student_model.pth')
    teacher_path = os.path.join(models_dir, 'teacher_model.pth')
    print('Current working directory:', os.getcwd())
    print('Looking for:', student_path, 'and', teacher_path)
    
    student_loaded = False
    teacher_loaded = False
    
    # Load Student
    student = VideoSharpeningStudent().to(device)
    if os.path.exists(student_path):
        print('Found student model weights!')
        try:
            result = student.load_state_dict(torch.load(student_path, map_location=device), strict=False)
            print('Student model load_state_dict result:', result)
            if len(result.missing_keys) == 0 and len(result.unexpected_keys) == 0:
                student_loaded = True
            else:
                print('WARNING: There are missing or unexpected keys in the student model weights!')
        except Exception as e:
            print(f'Error loading student weights: {e}')
    else:
        print('student_model.pth not found, using untrained student model.')
    student.eval()
    
    # Load Teacher
    teacher = ImageRestorationTeacher().to(device)
    if os.path.exists(teacher_path):
        print('Found teacher model weights!')
        try:
            result = teacher.load_state_dict(torch.load(teacher_path, map_location=device), strict=False)
            print('Teacher model load_state_dict result:', result)
            if len(result.missing_keys) == 0 and len(result.unexpected_keys) == 0:
                teacher_loaded = True
            else:
                print('WARNING: There are missing or unexpected keys in the teacher model weights!')
        except Exception as e:
            print(f'Error loading teacher weights: {e}')
    else:
        print('teacher_model.pth not found, using untrained teacher model.')
    teacher.eval()
    return student, teacher, student_loaded, teacher_loaded