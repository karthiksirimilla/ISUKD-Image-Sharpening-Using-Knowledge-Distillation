import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
import streamlit as st
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
from utils.model_loader import load_models
from utils.metrics import calculate_metrics
from utils.image_processing import (
    load_image, 
    preprocess_image, 
    postprocess_image,
    create_comparison
)
# Page configuration
st.set_page_config(
    page_title="ISUKD - Image Sharpening with Knowledge Distillation",
    page_icon="🔍",
    layout="wide"
)

@st.cache_resource
def load_cached_models():
    """Load models with proper error handling"""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        student, teacher, student_loaded, teacher_loaded = load_models(device)
        return student, teacher, device, student_loaded, teacher_loaded
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()

def main():
    st.title("Image Sharpening using Knowledge Distillation")
    st.markdown("""
    Enhance your video conferencing quality with our AI-powered image sharpening technology.
    This application uses a teacher-student knowledge distillation approach.
    """)
    
    # Initialize session state
    if 'results' not in st.session_state:
        st.session_state.results = None
    
    # Load models
    student, teacher, device, student_loaded, teacher_loaded = load_cached_models()
    
    # Sidebar controls
    st.sidebar.header("Settings")
    input_source = st.sidebar.radio(
        "Select input source:",
        ("Upload an image", "Use sample image", "Upload a video")
    )
    
    # Image/Video input handling
    img_np = None
    video_file = None
    if input_source == "Upload an image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            img_np = load_image(uploaded_file)
    elif input_source == "Upload a video":
        video_file = st.file_uploader("Upload a video file...", type=["mp4", "avi", "mov", "mkv"])
    else:
        # Create sample images directory if it doesn't exist
        sample_dir = "static/sample_images"
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
            st.info("Sample images directory created. Please add some sample images to test the application.")
        
        if os.path.exists(sample_dir):
            sample_images = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if sample_images:
                selected_sample = st.sidebar.selectbox("Select sample image:", sample_images)
                if selected_sample:
                    img_np = load_image(open(f"{sample_dir}/{selected_sample}", "rb"))
            else:
                st.warning("No sample images found. Please upload an image instead.")

    # Video processing section
    if video_file is not None:
        import tempfile
        st.video(video_file)
        if st.button("Process Video"):
            with st.spinner("Processing video, please wait..."):
                # Save uploaded video to a temp file
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(video_file.read())
                tfile.close()
                input_path = tfile.name
                # Open video
                cap = cv2.VideoCapture(input_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                # Output temp files
                out_student_path = tempfile.mktemp(suffix='_student.mp4')
                out_teacher_path = tempfile.mktemp(suffix='_teacher.mp4')
                out_student = cv2.VideoWriter(out_student_path, fourcc, fps, (width, height))
                out_teacher = cv2.VideoWriter(out_teacher_path, fourcc, fps, (width, height))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                pbar = st.progress(0, text="Processing frames...")
                processed_frames = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Preprocess
                    input_tensor = preprocess_image(frame_rgb, img_size=None).to(device)
                    # Resize tensor to original frame size
                    input_tensor = torch.nn.functional.interpolate(input_tensor, size=(height, width), mode='bilinear', align_corners=False)
                    # Model inference
                    with torch.no_grad():
                        student_output = student(input_tensor)
                        teacher_output = teacher(input_tensor)
                    # Postprocess
                    student_np = postprocess_image(student_output, target_size=(width, height))
                    teacher_np = postprocess_image(teacher_output, target_size=(width, height))
                    # Convert RGB to BGR for OpenCV
                    student_bgr = cv2.cvtColor(student_np, cv2.COLOR_RGB2BGR)
                    teacher_bgr = cv2.cvtColor(teacher_np, cv2.COLOR_RGB2BGR)
                    out_student.write(student_bgr)
                    out_teacher.write(teacher_bgr)
                    processed_frames += 1
                    pbar.progress(min(processed_frames / frame_count, 1.0), text=f"Processing frames... ({processed_frames}/{frame_count})")
                cap.release()
                out_student.release()
                out_teacher.release()
                pbar.empty()
                st.success(f"Video processing complete! Processed {processed_frames} frames.")
                # Download buttons
                with open(out_student_path, "rb") as f:
                    st.download_button("Download Student Output Video", f.read(), file_name="student_output.mp4", mime="video/mp4")
                with open(out_teacher_path, "rb") as f:
                    st.download_button("Download Teacher Output Video", f.read(), file_name="teacher_output.mp4", mime="video/mp4")
    
    if img_np is not None:
        # Display input image
        st.image(img_np, caption="Input Image", use_container_width=True, channels="RGB")
        
        if st.button("Enhance Image"):
            with st.spinner("Processing..."):
                try:
                    start_time = time.time()
                    original_size = img_np.shape[:2]  # Keep original dimensions
                    
                    # Preprocess
                    input_tensor = preprocess_image(img_np).to(device)
                    
                    # Model inference
                    with torch.no_grad():
                        student_output = student(input_tensor)
                        teacher_output = teacher(input_tensor)
                    
                    # Postprocess and resize to original dimensions
                    student_np = postprocess_image(student_output, target_size=(original_size[1], original_size[0]))
                    teacher_np = postprocess_image(teacher_output, target_size=(original_size[1], original_size[0]))
                    
                    # Overlay sharpening features from student output onto the original input
                    student_float = student_np.astype(np.float32)
                    input_float = img_np.astype(np.float32)
                    sharpen_features = student_float - input_float
                    alpha = 1.0  # You can tune this value for more/less sharpening
                    sharpened = input_float + alpha * sharpen_features
                    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
                    
                    # Color restoration: combine luminance from student output with color from input
                    # Ensure both are uint8 RGB
                    input_rgb = img_np.astype(np.uint8)
                    student_rgb = student_np.astype(np.uint8)
                    # Convert to YCrCb
                    input_ycc = cv2.cvtColor(input_rgb, cv2.COLOR_RGB2YCrCb)
                    student_ycc = cv2.cvtColor(student_rgb, cv2.COLOR_RGB2YCrCb)
                    # Replace Y channel
                    color_restored_ycc = input_ycc.copy()
                    color_restored_ycc[..., 0] = student_ycc[..., 0]
                    # Convert back to RGB
                    color_restored = cv2.cvtColor(color_restored_ycc, cv2.COLOR_YCrCb2RGB)

                    # Debug: print channel means
                    print('Student output shape:', student_np.shape)
                    print('Student output channel means:', student_np.mean(axis=(0,1)))
                    print('Color restored output channel means:', color_restored.mean(axis=(0,1)))

                    # Calculate metrics
                    student_metrics = calculate_metrics(img_np, student_np)
                    teacher_metrics = calculate_metrics(img_np, teacher_np)
                    comparison_metrics = calculate_metrics(teacher_np, student_np)
                    
                    # Store results
                    st.session_state.results = {
                        'input': img_np,
                        'student': student_np,
                        'teacher': teacher_np,
                        'color_restored': color_restored,
                        'metrics': {
                            'student': student_metrics,
                            'teacher': teacher_metrics,
                            'comparison': comparison_metrics
                        },
                        'time': time.time() - start_time
                    }
                    
                except Exception as e:
                    st.error(f"Processing failed: {str(e)}")
                    import traceback
                    st.error(f"Full error: {traceback.format_exc()}")
                    st.session_state.results = None
    
    # Display results if available
    if st.session_state.results:
        results = st.session_state.results
        
        st.success(f"Processing complete! Time taken: {results['time']:.2f}s")
        
        # Show correct info/success message based on weights loaded
        if not (student_loaded and teacher_loaded):
            st.info("""
            **Note**: The models are currently using untrained weights since no pre-trained model files were found or loaded successfully.
            The outputs may appear similar or grayish. For better results, please add trained model files
            (`student_model.pth` and `teacher_model.pth`) to the `models/` directory.
            """)
        else:
            st.success("✅ Pretrained model weights loaded successfully!")
        
        # Debug information
        with st.expander("Debug Information"):
            st.write(f"Input image shape: {results['input'].shape}")
            st.write(f"Student output shape: {results['student'].shape}")
            st.write(f"Teacher output shape: {results['teacher'].shape}")
            st.write(f"Input image dtype: {results['input'].dtype}")
            st.write(f"Student output dtype: {results['student'].dtype}")
            st.write(f"Teacher output dtype: {results['teacher'].dtype}")
            st.write(f"Student output range: [{results['student'].min()}, {results['student'].max()}]")
            st.write(f"Teacher output range: [{results['teacher'].min()}, {results['teacher'].max()}]")
        
        # Display individual images first
        st.subheader("Individual Results")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.write("**Input Image**")
            st.image(results['input'], caption="Original Input", use_container_width=True, channels="RGB")
        
        with col2:
            st.write("**Student Output**")
            student_display = results['student'].copy()
            if student_display.dtype != np.uint8:
                student_display = (student_display * 255).astype(np.uint8)
            st.image(student_display, caption="Student Model Output", use_container_width=True, channels="RGB")
        
        with col3:
            st.write("**Color Restored Output**")
            color_restored_display = results['color_restored'].copy()
            st.image(color_restored_display, caption="Sharpened Output (Color Restored)", use_container_width=True, channels="RGB")
        
        with col4:
            st.write("**Teacher Output**")
            teacher_display = results['teacher'].copy()
            if teacher_display.dtype != np.uint8:
                teacher_display = (teacher_display * 255).astype(np.uint8)
            st.image(teacher_display, caption="Teacher Model Output", use_container_width=True, channels="RGB")
        
        # Create comparison using matplotlib as backup
        st.subheader("Side-by-Side Comparison")
        try:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            titles = ["Input", "Student Output", "Teacher Output"]
            images = [results['input'], results['student'], results['teacher']]
            
            for ax, title, img in zip(axes, titles, images):
                # Ensure image is in correct format for matplotlib
                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8)
                ax.imshow(img)
                ax.set_title(title, fontsize=14, fontweight='bold')
                ax.axis('off')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)  # Close to prevent memory issues
        except Exception as e:
            st.error(f"Error displaying comparison: {str(e)}")
            # Fallback to simple display
            st.write("Comparison display failed. See individual results above.")
        
        # Metrics display
        st.subheader("Performance Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("PSNR (Student)", f"{results['metrics']['student']['psnr']:.2f} dB")
            st.metric("SSIM (Student)", f"{results['metrics']['student']['ssim']:.4f}")
        
        with col2:
            st.metric("PSNR (Teacher)", f"{results['metrics']['teacher']['psnr']:.2f} dB")
            st.metric("SSIM (Teacher)", f"{results['metrics']['teacher']['ssim']:.4f}")
        
        with col3:
            st.metric("PSNR Difference", 
                     f"{results['metrics']['teacher']['psnr'] - results['metrics']['student']['psnr']:.2f} dB")
            st.metric("SSIM Difference", 
                     f"{results['metrics']['teacher']['ssim'] - results['metrics']['student']['ssim']:.4f}")
        
        # Download buttons
        st.subheader("Download Results")
        col1, col2, col3 = st.columns(3)
        with col1:
            student_download = results['student'].copy()
            if student_download.dtype != np.uint8:
                student_download = (student_download * 255).astype(np.uint8)
            st.download_button(
                label="Download Student Output",
                data=cv2.imencode('.png', student_download)[1].tobytes(),
                file_name="student_output.png",
                mime="image/png"
            )
        with col2:
            color_restored_download = results['color_restored'].copy()
            st.download_button(
                label="Download Color Restored Output",
                data=cv2.imencode('.png', color_restored_download)[1].tobytes(),
                file_name="color_restored_output.png",
                mime="image/png"
            )
        with col3:
            teacher_download = results['teacher'].copy()
            if teacher_download.dtype != np.uint8:
                teacher_download = (teacher_download * 255).astype(np.uint8)
            st.download_button(
                label="Download Teacher Output",
                data=cv2.imencode('.png', teacher_download)[1].tobytes(),
                file_name="teacher_output.png",
                mime="image/png"
            )

if __name__ == "__main__":
    main()