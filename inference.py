from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from MonkeyOCR.magic_pdf.model.custom_model import MonkeyOCR
from MonkeyOCR.api.main import is_async_model



def model_setup(config_path: str)->MonkeyOCR:
    """Initialize MonkeyOCR model"""
    monkey_ocr_model = MonkeyOCR(config_path)
    supports_async = is_async_model(monkey_ocr_model)
    return monkey_ocr_model

def single_task_recognition(input_file: str, MonkeyOCR_model: MonkeyOCR, task: str)->str:
    """
    Single task recognition for specific content type
    
    Args:
        input_file: Input file path
        MonkeyOCR_model: Pre-initialized model instance
        task: Task type ('text', 'formula', 'table')
    """
    print(f"Starting single task recognition: {task}")
    print(f"Processing file: {input_file}")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file does not exist: {input_file}")

    # Get task instruction
    instruction = TASK_INSTRUCTIONS.get(task, TASK_INSTRUCTIONS['text'])

    # Check file type and prepare images
    file_extension = input_file.split(".")[-1].lower()
    images = []
    
    if file_extension == 'pdf':
        print("⚠️  WARNING: PDF input detected for single task recognition.")
        print("⚠️  WARNING: Converting all PDF pages to images for processing.")
        print("⚠️  WARNING: This may take longer and use more resources than image input.")
        print("⚠️  WARNING: Consider using individual images for better performance.")
        
        try:
            # Convert PDF pages to PIL images directly
            print("Converting PDF pages to images...")
            images = pdf_to_images(input_file)
            print(f"Converted {len(images)} pages to images")
            
        except Exception as e:
            raise RuntimeError(f"Failed to convert PDF to images: {str(e)}")
            
    elif file_extension in ['jpg', 'jpeg', 'png']:
        # Load single image
        from PIL import Image
        images = [Image.open(input_file)]
    else:
        raise ValueError(f"Single task recognition supports PDF and image files, got: {file_extension}")
    
    # Start recognition
    print(f"Performing {task} recognition on {len(images)} image(s)...")
    start_time = time.time()
    
    try:
        # Prepare instructions for all images
        instructions = [instruction] * len(images)
        
        # Use chat model for single task recognition with PIL images directly
        responses = MonkeyOCR_model.chat_model.batch_inference(images, instructions)
        
        recognition_time = time.time() - start_time
        print(f"Recognition time: {recognition_time:.2f}s")
        
        # Combine results
        combined_result = responses[0]
        for i, response in enumerate(responses):
            if i > 0:
                combined_result = combined_result + "\n\n" + response
        
        # Save result
        # result_filename = f"{name_without_suff}_{task}_result.md"
        # md_writer.write(result_filename, combined_result.encode('utf-8'))
        print(f"combined_result.encode('utf-8')")
        
        print(f"Single task recognition completed!")
        print(f"Task: {task}")
        print(f"Processed {len(images)} image(s)")
        # print(f"Result saved to: {os.path.join(local_md_dir, result_filename)}")
        
        # Clean up resources
        try:
            # Give some time for async tasks to complete
            time.sleep(0.5)
            
            # Close images if they were opened
            for img in images:
                if hasattr(img, 'close'):
                    img.close()
                    
        except Exception as cleanup_error:
            print(f"Warning: Error during cleanup: {cleanup_error}")
        
        return combined_result
        
    except Exception as e:
        raise RuntimeError(f"Single task recognition failed: {str(e)}")

