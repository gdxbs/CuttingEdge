import random
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

# --- Helper Functions ---

def create_organic_shape_mask(size, complexity, irregularity):
    """
    Generates a random, organic-looking shape to serve as a mask.
    """
    # 1. Start with a black image
    img = Image.new('L', size, 0)
    draw = ImageDraw.Draw(img)

    # 2. Define Polygon Vertices based on polar coordinates
    center = (size[0] // 2, size[1] // 2)
    max_radius = min(size) // 3 
    points = []
    
    for i in range(complexity):
        angle = (i / complexity) * 2 * np.pi
        r_variance = random.uniform(-irregularity, irregularity)
        a_variance = random.uniform(-irregularity/2, irregularity/2)
        
        r = max_radius * (1 + r_variance)
        a = angle + a_variance
        
        points.append((center[0] + r * np.cos(a), center[1] + r * np.sin(a)))

    # 3. Draw the filled white polygon
    draw.polygon(points, fill=255)
    
    # 4. Smooth the shape
    img = img.filter(ImageFilter.GaussianBlur(radius=size[0]//25))
    img = img.point(lambda x: 255 if x > 128 else 0, 'L')
    
    return img

def create_fabric_texture(size, type):
    """Generates a procedural texture based on the chosen fabric type."""
    img = Image.new('RGB', size)
    draw = ImageDraw.Draw(img)
    
    if type == 'denim':
        base_color = (random.randint(30, 60), random.randint(70, 100), random.randint(120, 160))
        img.paste(base_color, [0, 0, size[0], size[1]])
        noise = np.random.normal(0, 20, (size[1], size[0], 3)).astype(np.uint8)
        noise_img = Image.fromarray(noise, 'RGB')
        img = Image.blend(img, noise_img, 0.3)
        
    elif type == 'knit':
        base_val = random.randint(140, 170)
        base_color = (base_val, base_val, base_val)
        img.paste(base_color, [0, 0, size[0], size[1]])
        
        line_color = (base_val-30, base_val-30, base_val-30)
        spacing = random.randint(6, 12)
        for i in range(-size[1], size[0] + size[1], spacing):
            draw.line([(i, 0), (i + size[1], size[1])], fill=line_color, width=random.randint(2, 4))
            
    elif type == 'leather':
        base_val = random.randint(20, 40)
        base_color = (base_val, base_val, base_val)
        img.paste(base_color, [0, 0, size[0], size[1]])
        
        noise = np.random.normal(0, 8, (size[1], size[0], 3)).astype(np.uint8)
        noise_img = Image.fromarray(noise, 'RGB')
        img = Image.blend(img, noise_img, 0.2)
        
    return img


# --- Gemini API Integration ---
try:
    import google.generativeai as genai
    from dotenv import load_dotenv
    
    load_dotenv(dotenv_path=".env.local")
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google-generativeai or python-dotenv not installed.")

def generate_gemini_texture(size, cloth_type):
    """
    Attempts to generate a texture using Gemini/Imagen API.
    Returns PIL Image if successful, None otherwise.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not GEMINI_AVAILABLE or not api_key:
        return None
        
    try:
        genai.configure(api_key=api_key)
        
        # Using gemini-2.5-flash-image which supports generateContent for images
        model = genai.GenerativeModel("models/gemini-2.5-flash-image")
        
        prompt = (
            f"High quality, seamless texture of {cloth_type} fabric. "
            "Top down view, flat lighting, realistic detail, 4k resolution. "
            "Texture only, no borders."
        )
        
        response = model.generate_content(prompt)
        
        if response.parts:
            # We expect the last part to be the image if successful
            # Depending on the API behavior, it might be inline data.
            # Usually response.parts[0].inline_data
            for part in response.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                     # Access the image data
                    image_data = part.inline_data.data
                    import io
                    return Image.open(io.BytesIO(image_data))
                # Fallback for some SDK versions where it might be directly in 'image' attribute or similar
                # But 'inline_data' is standard for multimodal responses in this SDK.
                
            # If we didn't find inline_data, try checking if the response itself handles it
            # But the loop above covers the standard case.
            
    except Exception as e:
        print(f"Gemini API generation failed: {e}") 
        return None
    
    return None

# --- Helper Functions ---

def generate_single_cloth_image(index):
    IMG_SIZE = (800, 800)
    
    # 1. Create Background
    bg_color = (245, 242, 235)
    background = Image.new('RGB', IMG_SIZE, bg_color)
    bg_noise_arr = np.random.normal(0, 3, (IMG_SIZE[1], IMG_SIZE[0], 3)).astype(np.uint8)
    bg_noise = Image.fromarray(bg_noise_arr, 'RGB')
    background = Image.blend(background, bg_noise, 0.05)

    # 2. Define Cloth Parameters randomly
    cloth_type = random.choice(['leather', 'denim', 'knit'])
    shape_complexity = random.randint(8, 16) 
    shape_irregularity = random.uniform(0.3, 0.6)

    # 3. Generate Shape Mask
    mask = create_organic_shape_mask(IMG_SIZE, shape_complexity, shape_irregularity)
    
    # Edge Treatment (updated to use ModeFilter per your fix)
    if cloth_type == 'denim':
        mask = mask.filter(ImageFilter.ModeFilter(size=11))
        mask = mask.filter(ImageFilter.GaussianBlur(radius=1))

    # --- NEW: Calculate Dimensions ---
    # getbbox() returns (left, upper, right, lower) of the non-zero regions
    bbox = mask.getbbox()
    if bbox:
        cloth_w = bbox[2] - bbox[0]
        cloth_h = bbox[3] - bbox[1]
    else:
        # Fallback in unlikely case mask is empty
        cloth_w, cloth_h = 0, 0

    # 4. Generate Texture
    # Try Gemini API first
    texture_img = generate_gemini_texture(IMG_SIZE, cloth_type)
    
    if texture_img:
        print(f"Used Gemini API for {cloth_type}")
        texture_img = texture_img.resize(IMG_SIZE)
    else:
        # Fallback to procedural
        texture_img = create_fabric_texture(IMG_SIZE, cloth_type)
    
    # 5. Cut out the cloth
    cloth_layer = Image.new('RGBA', IMG_SIZE, (0,0,0,0))
    cloth_layer.paste(texture_img, (0,0), mask)

    # 6. Generate Drop Shadow
    shadow_mask = mask.filter(ImageFilter.GaussianBlur(radius=12))
    shadow_layer = Image.new('RGBA', IMG_SIZE, (0,0,0,0))
    shadow_draw = ImageDraw.Draw(shadow_layer)
    shadow_draw.rectangle([(0,0), IMG_SIZE], fill=(0,0,0,80))
    shadow_layer.putalpha(shadow_mask)

    # 7. Composite Final Image
    final_comp = background.copy()
    final_comp.paste(shadow_layer, (10, 15), shadow_layer)
    final_comp.paste(cloth_layer, (0, 0), cloth_layer)
    
    # 8. Save with Dynamic Name
    output_dir = "images/cloth/freeform"
    os.makedirs(output_dir, exist_ok=True)
    filename = f"cloth_{index}_{cloth_w}x{cloth_h}.jpg"
    output_path = os.path.join(output_dir, filename)
    final_comp.save(output_path)
    print(f"Generated {cloth_type} (Size: {cloth_w}x{cloth_h}) -> {output_path}")

# --- Execution Block ---
if __name__ == "__main__":
    # Updated to 50 as per your request
    num_images_to_generate = 50
    print(f"Starting generation of {num_images_to_generate} cloth images...")
    
    for i in range(num_images_to_generate):
        try:
            # We pass the index 'i+1' so the filename starts at 1
            generate_single_cloth_image(i + 1)
        except Exception as e:
            print(f"Error generating image {i+1}: {e}")
            
    print("Generation complete.")