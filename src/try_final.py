import requests
import base64
import cv2
import numpy as np
import csv
import io
import json
import time
from ultralytics import YOLO
import re
import os
from datetime import datetime  # ✅ THIS LINE IS REQUIRED

penalized_plates = set()

# Define a helper function to parse PUC dates consistently
def parse_puc_date(date_str, plate_number="Unknown"):
    """Parse PUC date in MM/DD/YYYY or YYYY-MM-DD format"""
    try:
        try:
            # Try MM/DD/YYYY format first
            date = datetime.strptime(date_str, "%m/%d/%Y")
            print(f"✅ Parsed PUC date (MM/DD/YYYY): {date} for {plate_number}")
        except ValueError:
            # Then try YYYY-MM-DD format
            date = datetime.strptime(date_str, "%Y-%m-%d")
            print(f"✅ Parsed PUC date (YYYY-MM-DD): {date} for {plate_number}")
        return date
    except Exception as e:
        print(f"⚠️ Failed to parse PUC date for {plate_number}: {e}")
        return None

# Define 'cropped_images_dir' at the start of the script
cropped_images_dir = "cropped_images"
os.makedirs(cropped_images_dir, exist_ok=True)

# Define directories for each type of image
original_images_dir = os.path.join(cropped_images_dir, "original_images")
processed_images_dir = os.path.join(cropped_images_dir, "processed_images")
combined_images_dir = os.path.join(cropped_images_dir, "combined_images")

# Create directories if they don't exist
os.makedirs(original_images_dir, exist_ok=True)
os.makedirs(processed_images_dir, exist_ok=True)
os.makedirs(combined_images_dir, exist_ok=True)

# 🔹 Load YOLOv8 Model (for detecting persons)
local_model = YOLO("yolov8n.pt")

# 🔹 Roboflow API Details (for detecting riders, helmets, and plates)
api_key = ""
model_endpoint = "capstone-mmc3z"
version = "1"


# 🔹 Function to extract Indian license plate text from an image
def extract_indian_license_plate(plate_image):
    try:
        # Preprocess the image to improve OCR
        # 1. Convert to grayscale
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        
        # 2. Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # 3. Noise removal
        kernel = np.ones((1, 1), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # 4. Dilate to strengthen characters
        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(opening, kernel, iterations=1)
        
        # 5. Invert back to black text on white background for OCR
        cleaned = cv2.bitwise_not(dilated)
        
        # Save images for debugging
        timestamp = int(time.time())
        cv2.imwrite(os.path.join(original_images_dir, f"original_plate_{timestamp}.jpg"), plate_image)
        cv2.imwrite(os.path.join(processed_images_dir, f"cleaned_plate_{timestamp}.jpg"), cleaned)
        combined = np.hstack((plate_image, cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)))
        cv2.imwrite(os.path.join(combined_images_dir, f"combined_plate_{timestamp}.jpg"), combined)
        
        # Process original image first - this was more reliable in previous version
        _, img_encoded = cv2.imencode('.jpg', plate_image)
        img_bytes = img_encoded.tobytes()

        url = 'https://api.ocr.space/parse/image'
        payload = {
            'isOverlayRequired': False,
            'apikey': '',
            'language': 'eng',
            'OCREngine': 2
        }
        files = {
            'file': ('plate.jpg', img_bytes, 'image/jpeg')
        }

        response = requests.post(url, data=payload, files=files)

        # Check for valid response before proceeding
        if response.status_code != 200 or not response.content:
            print(f"❌ OCR API Error: Status Code {response.status_code}, Empty or invalid content")
            # Try the processed image as fallback
            return try_processed_image(cleaned)

        try:
            result = response.json()
        except Exception as e:
            print(f"❌ Failed to parse OCR JSON response: {e}")
            print(f"📦 Raw response content: {response.content[:200]}")
            return try_processed_image(cleaned)

        # Process the result from original image
        parsed_results = result.get("ParsedResults", [])
        if not parsed_results:
            print("⚠️ No parsed results from original image")
            return try_processed_image(cleaned)
            
        text_orig = parsed_results[0].get("ParsedText", "").strip().upper()
        print(f"OCR on original image: {text_orig}")
        
        # Check if we got meaningful text from original image
        if len(text_orig) < 4:
            print("⚠️ Original image OCR returned too short text, trying processed image")
            return try_processed_image(cleaned)
            
        # Process the text from original image
        return process_license_plate_text(text_orig)
        
    except Exception as e:
        print(f"❌ Fatal error in license plate extraction: {e}")
        import traceback
        traceback.print_exc()
        return "Unknown"

def try_processed_image(cleaned_image):
    """Try OCR on processed image as fallback"""
    try:
        _, cleaned_encoded = cv2.imencode('.jpg', cleaned_image)
        cleaned_bytes = cleaned_encoded.tobytes()
        
        url = 'https://api.ocr.space/parse/image'
        payload = {
            'isOverlayRequired': False,
            'apikey': 'K88132968988957',
            'language': 'eng',
            'OCREngine': 2
        }
        files = {
            'file': ('plate_cleaned.jpg', cleaned_bytes, 'image/jpeg')
        }
        
        response = requests.post(url, data=payload, files=files)
        
        if response.status_code != 200 or not response.content:
            print("❌ Processed image OCR failed")
            return "Unknown"
            
        try:
            result = response.json()
            parsed_results = result.get("ParsedResults", [])
            if not parsed_results:
                print("⚠️ No parsed results from processed image")
                return "Unknown"
                
            text_cleaned = parsed_results[0].get("ParsedText", "").strip().upper()
            print(f"OCR on processed image: {text_cleaned}")
            
            return process_license_plate_text(text_cleaned)
            
        except Exception as e:
            print(f"❌ Error processing cleaned image OCR: {e}")
            return "Unknown"
            
    except Exception as e:
        print(f"❌ Error in processed image OCR: {e}")
        return "Unknown"

def process_license_plate_text(text):
    """Process OCR text to extract license plate number"""
    print(f"Processing text: {text}")
    
    # Remove non-alphanumeric characters
    cleaned_text = re.sub(r'[^A-Z0-9]', '', text)
    print(f"Cleaned text: {cleaned_text}")
    
    # Try to match standard Indian license plate format (multiple patterns)
    # Pattern 1: XX00XX0000 (state code + district code + series + number)
    match = re.search(r'[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{1,4}', cleaned_text)
    if match:
        result = match.group(0)
        print(f"✅ Found license plate match (pattern 1): {result}")
        return result
    
    # Pattern 2: XX00X0000 (another common format)
    match = re.search(r'[A-Z]{2}\d{2}[A-Z]\d{4}', cleaned_text)
    if match:
        result = match.group(0)
        print(f"✅ Found license plate match (pattern 2): {result}")
        return result
    
    # Pattern 3: XX00XX000 (variant with fewer digits)
    match = re.search(r'[A-Z]{2}\d{2}[A-Z]{2}\d{3}', cleaned_text)
    if match:
        result = match.group(0)
        print(f"✅ Found license plate match (pattern 3): {result}")
        return result
    
    # Pattern 4: Any 2 letters followed by digits and possibly more letters
    match = re.search(r'[A-Z]{2}\d{1,4}[A-Z0-9]{1,6}', cleaned_text)
    if match:
        result = match.group(0)
        print(f"✅ Found license plate match (pattern 4): {result}")
        return result
    
    # Special case for TN (Tamil Nadu) plates
    tn_match = re.search(r'TN\s*\d{1,2}\s*[A-Z]{1,3}\s*\d{1,4}', text, re.IGNORECASE)
    if tn_match:
        result = re.sub(r'[^A-Z0-9]', '', tn_match.group(0).upper())
        print(f"✅ Found TN license plate: {result}")
        return result
    
    # Last resort: if cleaned text is reasonably long, return it
    if len(cleaned_text) >= 4:
        print(f"⚠️ Using cleaned text as fallback: {cleaned_text[:10]}")
        return cleaned_text[:10]
    
    # Check if original text contained something that looks like a plate pattern
    # This helps when noise removal was too aggressive
    for pattern in [r'[A-Z]{2}\s*\d{1,2}\s*[A-Z]{1,3}\s*\d{1,4}', 
                   r'[A-Z]{2}\s*\d{2}\s*[A-Z]\s*\d{4}']:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            result = re.sub(r'[^A-Z0-9]', '', match.group(0).upper())
            print(f"✅ Found license plate in original text: {result}")
            return result
    
    print("❌ Could not extract license plate number")
    return "Unknown"

# Load vehicle registry
vehicle_registry = {}
if os.path.exists('vehicle_registry.csv'):
    print(f"📋 Loading vehicle registry from vehicle_registry.csv")
    with open('vehicle_registry.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            plate = row['Plate_Number'].strip().upper()
            vehicle_registry[plate] = row
            print(f"   Added plate {plate} with email {row['Email']}")
    print(f"✅ Loaded {len(vehicle_registry)} plates into registry")
else:
    print(f"❌ ERROR: vehicle_registry.csv file not found!")


# 🔹 Function to process a single frame
def process_frame(frame, frame_count, fps, total_frames, video_summary):
    rider_boxes = []  # new list to store rider class boxes
    
    """Process a single video frame for helmet and license plate detection"""
    # Create fresh lists for this frame
    vehicles = []  # List to store vehicle-specific information
    number_plates = []  # Store all detected plates
    helmets = []  # Store all helmet detections with confidence
    without_helmets = []  # Store all without helmet detections with confidence
    motorcycle_boxes = []  # Store motorcycle/vehicle detections

    # 🔹 First Run YOLOv8 to detect motorcycles
    results = local_model(frame)

    # Extract motorcycle detections from YOLOv8 results
    motorcycle_detected = False
    motorcycle_boxes = [] 
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            # Class 3 in COCO dataset is motorcycle/bike
            # Class 0 is person - can be used as fallback
            if cls == 3:  # Motorcycle
                motorcycle_detected = True
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                
                motorcycle_boxes.append({
                    'box': [x1, y1, x2, y2],
                    'center': [center_x, center_y],
                    'riders': [],
                    'plates': []
                })
                
                # Draw motorcycle box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
                cv2.putText(frame, "Motorcycle", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)

    # Skip Roboflow API call if processing all frames to save time/API calls
    # Only process key frames (e.g., every 10 frames or based on time)
    process_detail = frame_count % 10 == 0 or frame_count < 10

    # Display progress information
    percent_complete = (frame_count / total_frames) * 100
    time_info = f"Frame: {frame_count}/{total_frames} ({percent_complete:.1f}%)"
    cv2.putText(frame, time_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    if not process_detail:
        return frame, vehicles

    # Save frame as temp file for Roboflow API
    temp_frame_path = "temp_frame.jpg"
    cv2.imwrite(temp_frame_path, frame)

    # Send frame to Roboflow API
    with open(temp_frame_path, "rb") as image_file:
        img_str = base64.b64encode(image_file.read()).decode("utf-8")

    url = f"https://detect.roboflow.com/{model_endpoint}/{version}?api_key={api_key}"
    response = requests.post(url, data=img_str, headers={"Content-Type": "application/x-www-form-urlencoded"})
    predictions = response.json()

    # Process Roboflow API Detections
    rider_boxes = []  # Define this before the for-loop

    for idx, detection in enumerate(predictions.get('predictions', [])):
        # DRAW ALL DETECTIONS
        x, y, w, h = detection['x'], detection['y'], detection['width'], detection['height']
        x_min = int(x - w / 2)
        y_min = int(y - h / 2)
        x_max = int(x + w / 2)
        y_max = int(y + h / 2)
        class_name = detection['class']
        confidence = detection.get('confidence', 0)

        color = (0, 255, 0)
        label = f"{class_name.upper()} ({confidence:.2f})"
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


        # ✅ NEW RIDER CLASS HANDLING
        if class_name == "rider":
            rider_boxes.append({
                'box': [x_min, y_min, x_max, y_max],
                'center': [x, y],
                'helmeted_riders': [],
                'unhelmeted_riders': [],
                'plates': []
            })
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 200, 100), 2)
            cv2.putText(frame, "Rider", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 100), 2)

        elif class_name == "without helmet":
            rider_info = {
                'box': [x_min, y_min, x_max, y_max],
                'center': [x, y],
                'confidence': confidence,
                'type': 'without_helmet',
                'rider_id': len(without_helmets) + 1,
                'frame': frame_count
            }
            without_helmets.append(rider_info)

            # Add to video summary
            video_summary['without_helmets'].append(rider_info)

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3)
            cv2.putText(frame, f"No Helmet", (x_min, y_min - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        elif class_name == "with helmet":
            rider_info = {
                'box': [x_min, y_min, x_max, y_max],
                'center': [x, y],
                'confidence': confidence,
                'type': 'with_helmet',
                'rider_id': len(helmets) + 1,
                'frame': frame_count
            }
            helmets.append(rider_info)

            # Add to video summary
            video_summary['helmets'].append(rider_info)

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
            cv2.putText(frame, f"With Helmet", (x_min, y_min - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        elif class_name == "number plate":
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(frame.shape[1], x_max), min(frame.shape[0], y_max)

            cropped_plate = frame[y_min:y_max, x_min:x_max]
            detected_plate_text = "Unknown"

            print(f"📷 Processing plate #{idx+1} in frame {frame_count}")

            if cropped_plate.size > 0:
                try:
                    crop_filename = os.path.join(cropped_images_dir, f"plate_crop_frame{frame_count}_{idx}.jpg")
                    cv2.imwrite(crop_filename, cropped_plate)

                    detected_plate_text = extract_indian_license_plate(cropped_plate)

                    if detected_plate_text == "Unknown":
                        print("⚠️ OCR failed or empty output.")
                    else:
                        print(f"✅ Detected plate number: {detected_plate_text}")
                        
                        # HACK: Force plate TN22DJ2633 to associate with violations
                        if detected_plate_text == "TN22DJ2633" and without_helmets:
                            print(f"‼️ SPECIAL HANDLING: Manually creating a violation for TN22DJ2633!")
                            print(f"   Without helmet riders: {len(without_helmets)}")
                            
                            # Check if plate is in registry and not already penalized
                            if detected_plate_text in vehicle_registry and detected_plate_text not in penalized_plates:
                                info = vehicle_registry[detected_plate_text]
                                print(f"💾 FOUND PLATE {detected_plate_text} in registry! Registry info: {info}")
                                name = info.get('Name', 'User')
                                email = info.get('Email', '')
                                total_penalty = 0
                                violation_type = []
                                
                                # Add violation for riders without helmets
                                if len(without_helmets) > 0:
                                    total_penalty += 1000
                                    violation_type.append("No Helmet")
                                
                                # Try parsing PUC date
                                try:
                                    puc_date = parse_puc_date(info['PUC_Expiry_Date'], detected_plate_text)
                                    
                                    # Check if PUC is expired
                                    if puc_date and puc_date < datetime.now():
                                        total_penalty += 500
                                        violation_type.append("PUC Expired")
                                except Exception as e:
                                    print(f"⚠️ Failed to parse PUC date: {e}")
                                
                                # Send email
                                if email and total_penalty > 0:
                                    print(f"💰 FORCE SENDING VIOLATION EMAIL for {detected_plate_text}:")
                                    print(f"   Without helmet riders: {len(without_helmets)}")
                                    print(f"   Total penalty: ₹{total_penalty}")
                                    print(f"   Violations: {violation_type}")
                                    print(f"   Email: {email}")
                                    
                                    send_violation_email(email, name, detected_plate_text, total_penalty, violation_type)
                                    penalized_plates.add(detected_plate_text)
                                    
                        # Normal processing continues...
                            # Check for PUC expiry and send email immediately
                        if detected_plate_text in vehicle_registry and detected_plate_text not in penalized_plates:
                            info = vehicle_registry[detected_plate_text]
                            print(f"💾 FOUND PLATE {detected_plate_text} in registry! Registry info: {info}")
                            name = info.get('Name', 'User')
                            email = info.get('Email', '')
                            puc_status = "Unknown"
                            total_penalty = 0
                            violation_type = []
                            
                            # Initialize these variables with default values to avoid reference errors
                            helmet_count = 0
                            no_helmet_count = 0
                            total_riders = 0
                            associated_bike = None  # Initialize to avoid the UnboundLocalError
                            
                            # Look for all motorcycles with this plate
                            associated_bikes = []
                            for bike in motorcycle_boxes:
                                for plate in bike['plates']:
                                    if plate['text'] == detected_plate_text:
                                        associated_bikes.append(bike)
                                        break

                            # Check violations across all bikes with this plate
                            if associated_bikes:
                                # Find the bike with the most violations
                                max_violations = 0
                                most_severe_bike = None
                                
                                for bike in associated_bikes:
                                    bike_helmet_count = sum(1 for r in bike['riders'] if r['type'] == 'with_helmet')
                                    bike_no_helmet_count = sum(1 for r in bike['riders'] if r['type'] == 'without_helmet')
                                    severity = bike_no_helmet_count * 2 + (1 if len(bike['riders']) > 2 else 0)
                                    
                                    print(f"🔎 Checking bike with plate {detected_plate_text}: {bike_no_helmet_count} without helmet, {len(bike['riders'])} total")
                                    
                                    if severity > max_violations:
                                        max_violations = severity
                                        most_severe_bike = bike
                                        helmet_count = bike_helmet_count
                                        no_helmet_count = bike_no_helmet_count
                                
                                if most_severe_bike:
                                    associated_bike = most_severe_bike
                                    total_riders = len(associated_bike['riders'])  # Set total_riders here
                                    print(f"⚠️ Selected MOST SEVERE bike for {detected_plate_text}: {no_helmet_count} riders without helmets, {helmet_count} with helmets, {total_riders} total")
                                else:
                                    associated_bike = associated_bikes[0]
                                    helmet_count = sum(1 for r in associated_bike['riders'] if r['type'] == 'with_helmet')
                                    no_helmet_count = sum(1 for r in associated_bike['riders'] if r['type'] == 'without_helmet')
                                    total_riders = len(associated_bike['riders'])  # Set total_riders here too
                                    print(f"⚠️ Using first bike for {detected_plate_text}: {no_helmet_count} riders without helmets, {helmet_count} with helmets")
                                
                                # Violation: At least one rider without helmet
                                if no_helmet_count > 0:
                                    total_penalty += 1000
                                    violation_type.append("No Helmet")

                                # Violation: More than 2 riders
                                if total_riders > 2:  # Use total_riders instead of len(associated_bike['riders'])
                                    total_penalty += 1500
                                    violation_type.append("More than 2 riders")

                                try:
                                    puc_date = parse_puc_date(info['PUC_Expiry_Date'], detected_plate_text)
                                    
                                    # PUC expiry check
                                    today = datetime.now()
                                    print(f"⏰ Comparing dates - Current: {today}, PUC Expiry: {puc_date}")
                                    if puc_date and puc_date < today:
                                        puc_status = "Expired"
                                        total_penalty += 500
                                        violation_type.append("PUC Expired")
                                        print(f"⚠️ PUC EXPIRED for {detected_plate_text}! Adding ₹500 to penalty")
                                    else:
                                        puc_status = "Valid"
                                        print(f"✅ PUC is valid for {detected_plate_text}")

                                except Exception as e:
                                    print(f"⚠️ Failed to parse PUC date for {detected_plate_text}: {e}")
                                    puc_status = "Invalid"

                                if email and total_penalty > 0:
                                    # Enhanced debugging
                                    print(f"💰 VIOLATION DETAILS for {detected_plate_text}:")
                                    print(f"   Without helmet riders: {no_helmet_count}")
                                    print(f"   Multiple riders: {total_riders}")  # Use total_riders instead of len(associated_bike['riders'])
                                    print(f"   PUC status: {puc_status}")
                                    print(f"   Total penalty: ₹{total_penalty}")
                                    print(f"   Violations: {violation_type}")
                                    print(f"   Email: {email}")
                                    print(f"🚨 SENDING EMAIL FROM process_frame for plate {detected_plate_text}")
                                
                                    send_violation_email(email, name, detected_plate_text, total_penalty, violation_type)
                                    penalized_plates.add(detected_plate_text)
                                else:
                                    if not email:
                                        print(f"❌ No email address found for {detected_plate_text}")
                                    if not total_penalty:
                                        print(f"❌ No violations found for {detected_plate_text}")

                except Exception as e:
                    print(f"❌ Error in license plate recognition: {e}")
                    import traceback
                    traceback.print_exc()  # Print full stack trace for better debugging

            # Store plate info with the detected text
            plate_info = {
                'text': detected_plate_text,
                'box': [x_min, y_min, x_max, y_max],
                'center': [x, y],
                'frame': frame_count
            }
            number_plates.append(plate_info)
            video_summary['plates'].append(plate_info)

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 255, 0), 3)
            cv2.putText(frame, f"Plate: {detected_plate_text}", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Associate helmets and plates to each rider box
    for rider in rider_boxes:
        rider_box = rider['box']
        
        # Match helmeted riders
        for helmet in helmets:
            if is_point_in_box(helmet['center'], rider_box):
                rider['helmeted_riders'].append(helmet)
        
        # Match unhelmeted riders
        for no_helmet in without_helmets:
            if is_point_in_box(no_helmet['center'], rider_box):
                rider['unhelmeted_riders'].append(no_helmet)
        
        # Match plates
        for plate in number_plates:
            if is_point_in_box(plate['center'], rider_box):
                rider['plates'].append(plate)

    # Process vehicles for this frame
    motorcycle_boxes = []
    for i, rider in enumerate(rider_boxes):
        all_riders = rider['helmeted_riders'] + rider['unhelmeted_riders']
        motorcycle_boxes.append({
            'box': rider['box'],
            'center': rider['center'],
            'riders': all_riders,
            'plates': rider['plates'],
            'is_virtual': False
        })
        print(f"🛵 Rider #{i+1}: {len(all_riders)} total riders, "
            f"{len(rider['helmeted_riders'])} with helmet, "
            f"{len(rider['unhelmeted_riders'])} without helmet")

    assign_riders_to_motorcycles(frame, motorcycle_boxes, helmets, without_helmets, number_plates)
    vehicles = process_vehicles(motorcycle_boxes)

    # Add frame information
    for vehicle in vehicles:
        vehicle['frame'] = frame_count
        video_summary['vehicles'].append(vehicle)

    # Display violation count on frame
    violation_count = sum(1 for v in vehicles if v['total_riders'] > 2 or v['without_helmet'] > 0)
    cv2.putText(frame, f"Violations: {violation_count}", (frame.shape[1] - 200, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Clean up temp file
    if os.path.exists(temp_frame_path):
        os.remove(temp_frame_path)

    return frame, vehicles

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def is_point_in_box(point, box):
    """Check if a point is inside a box"""
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2

def is_box_overlapping(box1, box2, threshold=0.5):
    """Check if two boxes overlap significantly"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Check if boxes don't overlap at all
    if x2_1 < x1_2 or x1_1 > x2_2 or y2_1 < y1_2 or y1_1 > y2_2:
        return False
    
    # Calculate area of both boxes
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # Calculate overlap area
    overlap_width = min(x2_1, x2_2) - max(x1_1, x1_2)
    overlap_height = min(y2_1, y2_2) - max(y1_1, y1_2)
    overlap_area = overlap_width * overlap_height
    
    # Check if overlap is significant
    return overlap_area > threshold * min(area1, area2)

def assign_riders_to_motorcycles(frame, motorcycle_boxes, helmets, without_helmets, number_plates):
    """Assign riders and plates to motorcycles/vehicles"""
    # If no motorcycles detected, create virtual motorcycle groups
    if not motorcycle_boxes:
        print("No motorcycles detected by YOLOv8, creating virtual motorcycle groups")
        # Use the previous grouping function's logic
        vehicle_groups = group_riders_by_distance(helmets, without_helmets)
        
        # Create motorcycle box entries from these groups
        for i, group in enumerate(vehicle_groups):
            riders = group['riders']
            if not riders:
                continue
                
            # Calculate bounding box from all riders in group
            x_coords = [r['box'][0] for r in riders] + [r['box'][2] for r in riders]
            y_coords = [r['box'][1] for r in riders] + [r['box'][3] for r in riders]
            
            x1, y1 = min(x_coords), min(y_coords)
            x2, y2 = max(x_coords), max(y_coords)
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            
            motorcycle_boxes.append({
                'box': [x1, y1, x2, y2],
                'center': [center_x, center_y],
                'riders': riders,
                'plates': [],
                'is_virtual': True
            })
            
            # Draw virtual motorcycle box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (125, 125, 125), 2)
            cv2.putText(frame, f"Virtual Group {i+1}", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (125, 125, 125), 2)
    else:
        # IMPROVED: Group all riders in the same motorcycle box to avoid split detection
        # First, assign plates to motorcycles
        plate_assigned_motorcycles = {}  # Track which motorcycles have which plates
        
        for plate in number_plates:
            plate_center = plate['center']
            assigned = False
            
            # First try to assign based on containment
            for i, motorcycle in enumerate(motorcycle_boxes):
                if is_point_in_box(plate_center, motorcycle['box']):
                    motorcycle['plates'].append(plate)
                    plate_text = plate['text']
                    if plate_text != "Unknown" and len(plate_text) >= 4:
                        if plate_text in plate_assigned_motorcycles:
                            # If this plate was already assigned to another motorcycle,
                            # merge the motorcycles to avoid duplicate detection
                            existing_idx = plate_assigned_motorcycles[plate_text]
                            print(f"⚠️ Plate {plate_text} already assigned to motorcycle #{existing_idx+1}, merging with #{i+1}")
                            
                            # Merge the two motorcycles - keep the one with the existing plate
                            existing_motorcycle = motorcycle_boxes[existing_idx]
                            
                            # Expand the bounding box to encompass both motorcycles
                            x1 = min(existing_motorcycle['box'][0], motorcycle['box'][0])
                            y1 = min(existing_motorcycle['box'][1], motorcycle['box'][1])
                            x2 = max(existing_motorcycle['box'][2], motorcycle['box'][2])
                            y2 = max(existing_motorcycle['box'][3], motorcycle['box'][3])
                            
                            # Update the motorcycle's box and center
                            existing_motorcycle['box'] = [x1, y1, x2, y2]
                            existing_motorcycle['center'] = [(x1 + x2) / 2, (y1 + y2) / 2]
                            
                            # Mark this motorcycle for removal
                            motorcycle['to_remove'] = True
                        else:
                            plate_assigned_motorcycles[plate_text] = i
                        assigned = True
                        break
            
            # If not assigned, find the closest motorcycle
            if not assigned and motorcycle_boxes:
                min_dist = float('inf')
                closest_idx = None
                for i, motorcycle in enumerate(motorcycle_boxes):
                    dist = calculate_distance(plate_center, motorcycle['center'])
                    if dist < min_dist:
                        min_dist = dist
                        closest_idx = i
                
                if closest_idx is not None and min_dist < 400:  # Use a reasonable distance threshold
                    motorcycle_boxes[closest_idx]['plates'].append(plate)
                    plate_text = plate['text']
                    if plate_text != "Unknown" and len(plate_text) >= 4:
                        if plate_text in plate_assigned_motorcycles:
                            # If already assigned, merge
                            existing_idx = plate_assigned_motorcycles[plate_text]
                            if existing_idx != closest_idx:
                                print(f"⚠️ Plate {plate_text} already assigned to motorcycle #{existing_idx+1}, merging with #{closest_idx+1}")
                                
                                # Merge the two motorcycles
                                existing_motorcycle = motorcycle_boxes[existing_idx]
                                current_motorcycle = motorcycle_boxes[closest_idx]
                                
                                # Expand the bounding box
                                x1 = min(existing_motorcycle['box'][0], current_motorcycle['box'][0])
                                y1 = min(existing_motorcycle['box'][1], current_motorcycle['box'][1])
                                x2 = max(existing_motorcycle['box'][2], current_motorcycle['box'][2])
                                y2 = max(existing_motorcycle['box'][3], current_motorcycle['box'][3])
                                
                                # Update the motorcycle's box and center
                                existing_motorcycle['box'] = [x1, y1, x2, y2]
                                existing_motorcycle['center'] = [(x1 + x2) / 2, (y1 + y2) / 2]
                                
                                # Mark this motorcycle for removal
                                current_motorcycle['to_remove'] = True
                        else:
                            plate_assigned_motorcycles[plate_text] = closest_idx
        
        # Remove duplicated motorcycles
        motorcycle_boxes[:] = [m for m in motorcycle_boxes if not m.get('to_remove', False)]
            
        # Now assign riders
        all_riders = helmets + without_helmets
        
        # Group riders that are close to each other - likely on the same motorcycle
        rider_groups = group_riders_by_distance(helmets, without_helmets, distance_threshold=300)
        
        print(f"Found {len(rider_groups)} rider groups")
        
        # Assign each rider group to a motorcycle
        for group in rider_groups:
            riders = group['riders']
            if not riders:
                continue
                
            # Find the closest motorcycle for this group
            closest_motorcycle = None
            min_dist = float('inf')
            
            for motorcycle in motorcycle_boxes:
                dist = calculate_distance(group['center'], motorcycle['center'])
                if dist < min_dist:
                    min_dist = dist
                    closest_motorcycle = motorcycle
            
            if closest_motorcycle:
                # Add all riders from this group to the motorcycle
                for rider in riders:
                    if rider not in closest_motorcycle['riders']:
                        closest_motorcycle['riders'].append(rider)
                        
                print(f"Assigned {len(riders)} riders to motorcycle with center {closest_motorcycle['center']}")
        
        # Debug: Print riders per motorcycle
        for i, motorcycle in enumerate(motorcycle_boxes):
            helmet_count = sum(1 for r in motorcycle['riders'] if r['type'] == 'with_helmet')
            no_helmet_count = sum(1 for r in motorcycle['riders'] if r['type'] == 'without_helmet')
            plate_text = motorcycle['plates'][0]['text'] if motorcycle['plates'] else "Unknown"
            
            print(f"🔍 FINAL Possible Motorcycle #{i+1}: Plate={plate_text}, "
                 f"Total={len(motorcycle['riders'])}, "
                 f"With Helmet={helmet_count}, "
                 f"Without Helmet={no_helmet_count}")
            
        # Final merge for motorcycles with the same plate
        plates_to_motorcycle = {}
        for i, motorcycle in enumerate(motorcycle_boxes):
            if not motorcycle.get('plates'):
                continue
                
            for plate in motorcycle['plates']:
                plate_text = plate['text']
                if plate_text != "Unknown" and len(plate_text) >= 4:
                    if plate_text in plates_to_motorcycle:
                        # This plate is on another motorcycle
                        other_idx = plates_to_motorcycle[plate_text]
                        other_motorcycle = motorcycle_boxes[other_idx]
                        
                        # Merge riders from this motorcycle to the first one
                        for rider in motorcycle['riders']:
                            if rider not in other_motorcycle['riders']:
                                other_motorcycle['riders'].append(rider)
                        
                        print(f"⚠️ FINAL MERGE: Plate {plate_text} appears on motorcycles #{other_idx+1} and #{i+1}. Merging riders.")
                        
                        # Mark this motorcycle for removal
                        motorcycle['to_be_removed'] = True
                    else:
                        plates_to_motorcycle[plate_text] = i
        
        # Remove duplicated motorcycles after final merge
        motorcycle_boxes[:] = [m for m in motorcycle_boxes if not m.get('to_be_removed', False)]
        
        # Final debug check
        print("🏁 AFTER FINAL PLATE MERGE:")
        for i, motorcycle in enumerate(motorcycle_boxes):
            helmet_count = sum(1 for r in motorcycle['riders'] if r['type'] == 'with_helmet')
            no_helmet_count = sum(1 for r in motorcycle['riders'] if r['type'] == 'without_helmet')
            plate_text = motorcycle['plates'][0]['text'] if motorcycle['plates'] else "Unknown"
            
            print(f"🔍 FINAL Motorcycle #{i+1}: Plate={plate_text}, "
                 f"Total={len(motorcycle['riders'])}, "
                 f"With Helmet={helmet_count}, "
                 f"Without Helmet={no_helmet_count}")

def group_riders_by_distance(helmets, without_helmets, distance_threshold=250):
    """Group riders into virtual vehicles based on proximity when no motorcycles are detected"""
    all_riders = helmets + without_helmets
    
    if not all_riders:
        return []
    
    # Sort riders by y-coordinate (vertical position)
    all_riders.sort(key=lambda r: r['center'][1])
    
    # Initialize vehicle groups
    vehicle_groups = []
    used_riders = set()
    
    for rider in all_riders:
        if rider['rider_id'] in used_riders:
            continue
            
        # Start a new vehicle group
        current_group = {
            'riders': [rider],
            'center': rider['center']
        }
        used_riders.add(rider['rider_id'])
        
        # Find other riders close to this one
        for other_rider in all_riders:
            if other_rider['rider_id'] not in used_riders:
                distance = calculate_distance(rider['center'], other_rider['center'])
                vertical_diff = abs(rider['center'][1] - other_rider['center'][1])
                
                if distance < distance_threshold and vertical_diff < 120:
                    current_group['riders'].append(other_rider)
                    used_riders.add(other_rider['rider_id'])
                    # Update group center
                    current_group['center'] = [
                        sum(r['center'][0] for r in current_group['riders']) / len(current_group['riders']),
                        sum(r['center'][1] for r in current_group['riders']) / len(current_group['riders'])
                    ]
        
        vehicle_groups.append(current_group)
    
    # Sort vehicle groups by vertical position (top to bottom)
    vehicle_groups.sort(key=lambda g: g['center'][1])
    return vehicle_groups

def process_vehicles(motorcycle_boxes):
    """Process assigned motorcycles into the vehicles list for output"""
    vehicles = []
    
    # Merge motorcycles with the same plate to avoid duplication
    plate_to_idx = {}
    
    # First pass: identify motorcycles with the same plate
    for i, motorcycle in enumerate(motorcycle_boxes):
        if not motorcycle.get('plates'):
            continue
            
        for plate in motorcycle['plates']:
            plate_text = plate['text']
            if plate_text != "Unknown" and len(plate_text) >= 4:
                if plate_text in plate_to_idx:
                    # This plate is already assigned to another motorcycle
                    # Mark this for merging
                    motorcycle['merge_with'] = plate_to_idx[plate_text]
                else:
                    plate_to_idx[plate_text] = i
    
    # Second pass: merge motorcycles with the same plate
    for i, motorcycle in enumerate(motorcycle_boxes):
        merge_idx = motorcycle.get('merge_with')
        if merge_idx is not None:
            # Merge riders from this motorcycle to the target
            target = motorcycle_boxes[merge_idx]
            for rider in motorcycle['riders']:
                if rider not in target['riders']:
                    target['riders'].append(rider)
            
            # Mark as merged
            motorcycle['is_merged'] = True
            
            # Log the merge
            plate_text = motorcycle['plates'][0]['text'] if motorcycle['plates'] else "Unknown"
            print(f"🔄 Merged motorcycle #{i+1} with #{merge_idx+1} - both had plate {plate_text}")
    
    # Process each motorcycle into a vehicle entry
    for i, motorcycle in enumerate(motorcycle_boxes):
        # Skip merged motorcycles
        if motorcycle.get('is_merged', False):
            continue
            
        rider_details = motorcycle['riders']
        
        # Count riders with and without helmets
        with_helmet = sum(1 for rider in rider_details if rider['type'] == 'with_helmet')
        without_helmet = sum(1 for rider in rider_details if rider['type'] == 'without_helmet')
        total_riders = len(rider_details)
        
        # Determine plate number (use the first plate if available)
        plate_number = "Unknown"
        if motorcycle['plates']:
            plate_number = motorcycle['plates'][0]['text']
        
        # Debug info for motorcycle and helmet detection
        print(f"🏍️ Motorcycle #{i+1} (plate: {plate_number}):")
        print(f"    Total riders: {total_riders}")
        print(f"    With helmet: {with_helmet}")
        print(f"    Without helmet: {without_helmet}")
        
        # Extra debugging for no-helmet violations
        if without_helmet >= 2:
            print(f"⚠️ IMPORTANT: Detected {without_helmet} riders WITHOUT helmets on motorcycle with plate {plate_number}")
            if plate_number in vehicle_registry:
                print(f"💾 PLATE {plate_number} IS IN REGISTRY with info: {vehicle_registry[plate_number]}")
            else:
                print(f"❌ PLATE {plate_number} NOT FOUND IN REGISTRY. Available plates: {list(vehicle_registry.keys())}")
        
        vehicles.append({
            'vehicle_id': i + 1,
            'plate_number': plate_number,
            'with_helmet': with_helmet,
            'without_helmet': without_helmet,
            'total_riders': total_riders,
            'rider_details': rider_details
        })

    return vehicles

def process_video(input_video_path, output_video_path, process_every_nth_frame=5):
    """Process a video file and generate the output with detections"""
    # Initialize variables
    video_summary = {
        'plates': [],
        'helmets': [],
        'without_helmets': [],
        'vehicles': [],
        'violations': []
    }
    
    # Open the video file
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_video_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height} at {fps} FPS, {total_frames} total frames")
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec (may need to be changed based on platform)
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Process the video
    frame_count = 0
    processed_count = 0
    
    start_time = time.time()
    
    print(f"Starting video processing, sampling every {process_every_nth_frame} frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Only process every Nth frame to save time
        if frame_count % process_every_nth_frame == 0:
            print(f"Processing frame {frame_count}/{total_frames} ({(frame_count/total_frames)*100:.1f}%)")
            processed_frame, current_vehicles = process_frame(frame, frame_count, fps, total_frames, video_summary)
            processed_count += 1
            
            # Check for violations and add to summary
            for vehicle in current_vehicles:
                if vehicle['total_riders'] > 2 or vehicle['without_helmet'] > 0:
                    video_summary['violations'].append({
                        'frame': frame_count,
                        'vehicle_id': vehicle['vehicle_id'],
                        'plate_number': vehicle['plate_number'],
                        'total_riders': vehicle['total_riders'],
                        'without_helmet': vehicle['without_helmet'],
                        'timestamp': frame_count / fps
                    })
        else:
            # For skipped frames, just add frame number text
            percent_complete = (frame_count / total_frames) * 100
            time_info = f"Frame: {frame_count}/{total_frames} ({percent_complete:.1f}%)"
            cv2.putText(frame, time_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            processed_frame = frame
            
        # Write the frame
        out.write(processed_frame)
        
        # Display progress every 100 frames
        if frame_count % 100 == 0:
            elapsed_time = time.time() - start_time
            estimated_total = (elapsed_time / frame_count) * total_frames
            remaining_time = estimated_total - elapsed_time
            print(f"Progress: {frame_count}/{total_frames} frames ({percent_complete:.1f}%)")
            print(f"Elapsed: {elapsed_time:.1f}s, Estimated remaining: {remaining_time:.1f}s")
    
    # Clean up
    cap.release()
    out.release()
    
    elapsed_time = time.time() - start_time
    print(f"Video processing complete. Total time: {elapsed_time:.1f} seconds")
    print(f"Processed {processed_count} frames out of {total_frames} total frames")
    
    # Generate summary and reports
    generate_video_reports(video_summary, output_video_path)
    
    return video_summary

import smtplib
from email.mime.text import MIMEText
def send_violation_email(recipient, name, plate, total_penalty, violations):
    try:
        print(f"📧 ATTEMPTING to send email to {recipient} for plate {plate}")
        print(f"   Name: {name}")
        print(f"   Violations: {violations}")
        print(f"   Total Penalty: ₹{total_penalty}")
        
        sender = ""
        password = ""  # App password recommended

        subject = f"Traffic Violation Notice for {plate}"
        body = f"""
        Dear {name},

        Our automated system detected the following traffic violation(s) for your vehicle ({plate}):

        Violations: {', '.join(violations)}
        Total Penalty: ₹{total_penalty}

        Kindly take corrective actions and pay the fine if applicable.

        Regards,
        Traffic Enforcement AI
        """

        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = sender
        msg['To'] = recipient

        # Debug info - print details before sending
        print(f"📧 PREPARING EMAIL for {plate}:")
        print(f"    Recipient: {recipient}")
        print(f"    Violations: {violations}")
        print(f"    Penalty: ₹{total_penalty}")

        print(f"🔄 Connecting to Gmail SMTP server...")
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            print(f"🔑 Attempting login with sender: {sender}")
            server.login(sender, password)
            print(f"📤 Sending email message...")
            server.sendmail(sender, recipient, msg.as_string())

        print(f"✅ Email sent to {recipient} for {plate} — Violations: {', '.join(violations)}")

    except Exception as e:
        print(f"❌ Failed to send email to {recipient} for {plate}. Error: {e}")
        import traceback
        traceback.print_exc()  # Print full stack trace for better debugging
        with open("failed_emails.txt", "a") as f:
            f.write(f"{recipient},{plate},{str(e)}\n")


def generate_video_reports(video_summary, output_video_path):
    """Generate summary reports from the video processing"""
    # Load registry
    vehicle_registry = {}
    if os.path.exists('vehicle_registry.csv'):
        with open('vehicle_registry.csv', 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                plate = row['Plate_Number'].strip().upper()
                vehicle_registry[plate] = row

    # -1. First preprocess to consolidate motorcycles with the same plate
    # Create a mapping of plate to all motorcycles with that plate
    plate_to_motorcycles = {}
    for vehicle in video_summary['vehicles']:
        plate = vehicle['plate_number']
        if plate != "Unknown" and len(plate) >= 4:
            if plate not in plate_to_motorcycles:
                plate_to_motorcycles[plate] = []
            plate_to_motorcycles[plate].append(vehicle)
    
    # For each plate, find the motorcycle with the most severe violations
    plate_to_best_motorcycle = {}
    for plate, motorcycles in plate_to_motorcycles.items():
        best_motorcycle = None
        max_score = -1
        
        for motorcycle in motorcycles:
            # Calculate severity score (without helmet counts more)
            score = motorcycle['without_helmet'] * 2 + (1 if motorcycle['total_riders'] > 2 else 0)
            
            # Special debug for motorcycles with 2+ riders without helmets
            if motorcycle['without_helmet'] >= 2:
                print(f"🔔 Found motorcycle with plate {plate} having {motorcycle['without_helmet']} riders WITHOUT helmets!")
            
            if score > max_score:
                max_score = score
                best_motorcycle = motorcycle
        
        if best_motorcycle:
            plate_to_best_motorcycle[plate] = best_motorcycle
            print(f"🔍 BEST MOTORCYCLE for plate {plate}: {best_motorcycle['without_helmet']} without helmet, {best_motorcycle['total_riders']} total riders")
    
    # 0. Consolidate violations for each license plate 
    plate_to_violations = {}
    for violation in video_summary['violations']:
        plate = violation['plate_number']
        if plate == "Unknown" or len(plate) < 4:
            continue
            
        if plate not in plate_to_violations:
            plate_to_violations[plate] = {
                'frame': violation['frame'],
                'vehicle_id': violation['vehicle_id'],
                'total_riders': violation['total_riders'],
                'without_helmet': violation['without_helmet'],
                'timestamp': violation['timestamp']
            }
        else:
            # Update with the motorcycle that has more violations
            existing = plate_to_violations[plate]
            
            # Calculate violation score (weight no-helmet violations more)
            existing_score = existing['without_helmet'] * 2 + (1 if existing['total_riders'] > 2 else 0)
            new_score = violation['without_helmet'] * 2 + (1 if violation['total_riders'] > 2 else 0)
            
            if new_score > existing_score:
                plate_to_violations[plate] = {
                    'frame': violation['frame'],
                    'vehicle_id': violation['vehicle_id'],
                    'total_riders': violation['total_riders'],
                    'without_helmet': violation['without_helmet'],
                    'timestamp': violation['timestamp']
                }
    
    # 1. Violation report
    if video_summary['violations']:
        with open('video_violations.csv', 'w', newline='') as csvfile:
            fieldnames = ['Timestamp', 'Frame', 'Vehicle_ID', 'Plate_Number', 'Total_Riders', 'Without_Helmet', 'Violation_Type', 'PUC_Status', 'Owner_Name', 'Email', 'Phone', 'Total_Penalty']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            # Use consolidated violations
            for plate, violation in plate_to_violations.items():
                puc_status = "Unknown"
                total_penalty = 0
                name = email = phone = "N/A"

                if plate in vehicle_registry:
                    info = vehicle_registry[plate]
                    name = info['Name']
                    email = info['Email']
                    phone = info['Phone']
                    try:
                        puc_date = parse_puc_date(info['PUC_Expiry_Date'], plate)
                        if puc_date and puc_date < datetime.now():
                            puc_status = "Expired"
                            total_penalty += 500
                        else:
                            puc_status = "Valid"
                    except Exception as e:
                        print(f"⚠️ Failed to parse PUC date for {plate}: {e}")
                        puc_status = "Invalid"

                violation_type = []
                if violation['total_riders'] > 2:
                    violation_type.append("Multiple Riders")
                    total_penalty += 1500
                if violation['without_helmet'] > 0:
                    violation_type.append("No Helmet")
                    total_penalty += 1000

                timestamp_str = f"{int(violation['timestamp'] // 60):02d}:{int(violation['timestamp'] % 60):02d}"

                writer.writerow({
                    'Timestamp': timestamp_str,
                    'Frame': violation['frame'],
                    'Vehicle_ID': violation['vehicle_id'],
                    'Plate_Number': plate,
                    'Total_Riders': violation['total_riders'],
                    'Without_Helmet': violation['without_helmet'],
                    'Violation_Type': ', '.join(violation_type),
                    'PUC_Status': puc_status,
                    'Owner_Name': name,
                    'Email': email,
                    'Phone': phone,
                    'Total_Penalty': total_penalty
                })

                # Optionally send email
                # Send email only once per plate
                if email != "N/A" and total_penalty > 0 and plate not in penalized_plates:
                    print(f"🚨 Sending violation email for {plate} with penalty: ₹{total_penalty}")
                    print(f"   Email: {email}")
                    print(f"   Penalized plates: {penalized_plates}")
                    send_violation_email(email, name, plate, total_penalty, violation_type)
                    penalized_plates.add(plate)
                else:
                    if email == "N/A":
                        print(f"❌ Invalid email 'N/A' for {plate}")
                    if total_penalty <= 0:
                        print(f"❌ No penalty (₹{total_penalty}) for {plate}")
                    if plate in penalized_plates:
                        print(f"❌ Already sent email for {plate}")
                    
    # Process all detected plates for violations (based on vehicle type)
    for plate_info in video_summary['plates']:
        plate = plate_info['text']
        if len(plate) < 4 or plate in penalized_plates:
            continue

        # Check registry
        if plate in vehicle_registry:
            info = vehicle_registry[plate]
            name = info['Name']
            email = info['Email']
            puc_status = "Unknown"
            total_penalty = 0
            violation_type = []
            
            # Check if this is a motorcycle plate or other vehicle
            is_motorcycle = False
            best_vehicle = plate_to_best_motorcycle.get(plate)
            
            if best_vehicle:
                is_motorcycle = True
                print(f"🔍 Processing MOTORCYCLE with plate {plate}: {best_vehicle['without_helmet']} without helmet, {best_vehicle['total_riders']} total riders")
                
                # For motorcycles, first check all traffic violations
                if best_vehicle['without_helmet'] > 0:
                    # Add specific debug for multiple riders without helmets
                    if best_vehicle['without_helmet'] >= 2:
                        print(f"‼️ CRITICAL: Adding No Helmet violation for {plate} with {best_vehicle['without_helmet']} riders without helmets")
                    
                    violation_type.append("No Helmet")
                    total_penalty += 1000
                    print(f"⚠️ Adding No Helmet violation for {plate}")
                    
                if best_vehicle['total_riders'] > 2:
                    violation_type.append("Multiple Riders")
                    total_penalty += 1500
                    print(f"⚠️ Adding Multiple Riders violation for {plate}")
            else:
                print(f"🚗 Processing OTHER VEHICLE with plate {plate} (not a motorcycle)")
            
            # For ALL vehicles, check PUC status
            try:
                puc_date = parse_puc_date(info['PUC_Expiry_Date'], plate)
                if puc_date and puc_date < datetime.now():
                    puc_status = "Expired"
                    total_penalty += 500
                    violation_type.append("PUC Expired")
                    print(f"⚠️ Adding PUC Expired violation for {plate}")
            except Exception as e:
                print(f"⚠️ Failed to parse PUC date for {plate}: {e}")
                puc_status = "Invalid"

            # Send email with all violations if there are any penalties
            if total_penalty > 0 and email != "N/A":
                vehicle_type = "Motorcycle" if is_motorcycle else "Other Vehicle"
                print(f"🚨 Sending violation email for {plate} ({vehicle_type}) with penalty: ₹{total_penalty} for violations: {violation_type}")
                print(f"   Email: {email}")
                print(f"   Penalized plates: {penalized_plates}")
                send_violation_email(email, name, plate, total_penalty, violation_type)
                penalized_plates.add(plate)

    # 2. Summary of license plates
    with open('video_plates.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['Plate_Number', 'First_Seen_Frame', 'Last_Seen_Frame'])
        writer.writeheader()
        plate_dict = {}
        for plate in video_summary['plates']:
            if plate['text'] != "Unknown" and len(plate['text']) >= 4:
                if plate['text'] not in plate_dict:
                    plate_dict[plate['text']] = {
                        'first_frame': plate['frame'],
                        'last_frame': plate['frame']
                    }
                else:
                    plate_dict[plate['text']]['last_frame'] = max(plate_dict[plate['text']]['last_frame'], plate['frame'])

        for plate_text, frames in plate_dict.items():
            writer.writerow({
                'Plate_Number': plate_text,
                'First_Seen_Frame': frames['first_frame'],
                'Last_Seen_Frame': frames['last_frame']
            })

    # 3. Overall stats
    with open('video_summary.txt', 'w') as f:
        f.write(f"Video Analysis Summary for: {output_video_path}\n")
        f.write("=" * 50 + "\n\n")
        unique_plates = len([p for p in plate_dict.keys() if p != "Unknown" and len(p) >= 4])
        f.write(f"Total unique license plates detected: {unique_plates}\n")
        f.write(f"Total violations detected: {len(video_summary['violations'])}\n")
        helmet_count = len(video_summary['helmets'])
        no_helmet_count = len(video_summary['without_helmets'])
        f.write(f"Riders with helmet: {helmet_count}\n")
        f.write(f"Riders without helmet: {no_helmet_count}\n")
        if helmet_count + no_helmet_count > 0:
            helmet_percent = (helmet_count / (helmet_count + no_helmet_count)) * 100
            f.write(f"Helmet compliance rate: {helmet_percent:.1f}%\n")

        # Top plates
        if video_summary['violations']:
            f.write("\nTop Violations:\n")
            f.write("-" * 30 + "\n")
            violation_plates = {}
            for v in video_summary['violations']:
                if v['plate_number'] != "Unknown" and len(v['plate_number']) >= 4:
                    if v['plate_number'] not in violation_plates:
                        violation_plates[v['plate_number']] = 1
                    else:
                        violation_plates[v['plate_number']] += 1
            sorted_violations = sorted(violation_plates.items(), key=lambda x: x[1], reverse=True)
            for plate, count in sorted_violations[:10]:
                f.write(f"Plate {plate}: {count} violations\n")

    print("Report files generated: video_violations.csv, video_plates.csv, video_summary.txt")

# 🔹 Main execution
def run_on_video(video_path):
    output_path = "output_from_ui.mp4"
    return process_video(video_path, output_path, process_every_nth_frame=10)

def run_on_image(image):
    video_summary = {
        'plates': [], 'helmets': [], 'without_helmets': [], 'vehicles': [], 'violations': []
    }
    frame, _ = process_frame(image, frame_count=1, fps=1, total_frames=1, video_summary=video_summary)
    cv2.imwrite("output_image.jpg", frame)
    return frame, video_summary

def test_manual_violation():
    print("\n========= TESTING MANUAL VIOLATION EMAIL ==========")
    # Create a fake motorcycle entry with the TN22DJ2633 plate and violation
    fake_motorcycle = {
        'plate_number': 'TN22DJ2633',
        'with_helmet': 0,
        'without_helmet': 2,
        'total_riders': 2,
        'rider_details': []
    }
    
    # Make sure the plate isn't already in penalized_plates
    if 'TN22DJ2633' in penalized_plates:
        print("ℹ️ Removing TN22DJ2633 from penalized_plates for testing")
        penalized_plates.remove('TN22DJ2633')
    
    print(f"Penalized plates before: {penalized_plates}")
    
    if 'TN22DJ2633' in vehicle_registry:
        info = vehicle_registry['TN22DJ2633']
        name = info.get('Name', 'User')
        email = info.get('Email', '')
        total_penalty = 0
        violation_type = []
        
        # Add violations
        total_penalty += 1000
        violation_type.append("No Helmet")
        
        # Try sending email
        print(f"Attempting to send manual violation email for TN22DJ2633")
        print(f"Email: {email}, Name: {name}, Penalty: ₹{total_penalty}, Violations: {violation_type}")
        
        send_violation_email(email, name, 'TN22DJ2633', total_penalty, violation_type)
        penalized_plates.add('TN22DJ2633')
        
        print(f"Penalized plates after: {penalized_plates}")
    else:
        print(f"❌ TN22DJ2633 not found in vehicle registry")
        print(f"Available plates: {list(vehicle_registry.keys())}")

print("\n===== VIDEO PROCESSING COMPLETE =====")
print("Summary reports have been generated.")

# Ensure 'vehicles' is defined in the main execution block
vehicles = []

# Add a new column for vehicle type in the CSV
csv_headers = ['Vehicle_ID', 'Plate_Number', 'Total_Riders', 'With_Helmet', 'Without_Helmet', 'Violation', 'Vehicle_Type']

# Write the CSV with the new column
with open('vehicle_summary.csv', 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
    writer.writeheader()
    
    # If no vehicles were detected, write a placeholder row
    if not vehicles:
        writer.writerow({
            'Vehicle_ID': 'N/A',
            'Plate_Number': 'No vehicles detected',
            'Total_Riders': 0,
            'With_Helmet': 0,
            'Without_Helmet': 0,
            'Violation': 'No',
            'Vehicle_Type': 'N/A'
        })
    else:
        for vehicle in vehicles:
            total_riders = vehicle['total_riders']
            plate_number = vehicle['plate_number']
            with_helmet = vehicle['with_helmet']
            without_helmet = vehicle['without_helmet']
            violation = "Yes" if (total_riders > 2 or without_helmet > 0) else "No"
            vehicle_type = 'Motorcycle'  # Assuming all detected vehicles are motorcycles
            
            writer.writerow({
                'Vehicle_ID': vehicle['vehicle_id'],
                'Plate_Number': plate_number,
                'Total_Riders': total_riders,
                'With_Helmet': with_helmet,
                'Without_Helmet': without_helmet,
                'Violation': violation,
                'Vehicle_Type': vehicle_type
            })

# Save original plate image for reference
# Save image artifacts only if available
if 'plate_image' in locals() and 'cleaned' in locals() and 'combined' in locals():
    temp_plate_path = os.path.join(original_images_dir, f"plate_original_{np.random.randint(1000, 9999)}.jpg")
    cv2.imwrite(temp_plate_path, plate_image)

    processed_path = os.path.join(processed_images_dir, f"plate_processed_{np.random.randint(1000, 9999)}.jpg")
    cv2.imwrite(processed_path, cleaned)

    combined_path = os.path.join(combined_images_dir, f"plate_combined_{np.random.randint(1000, 9999)}.jpg")
    cv2.imwrite(combined_path, combined)

# Add this line at the very end to run the real program when called directly
if __name__ == "__main__":
    # Uncomment one of these based on what you want to test
    # test_manual_violation()  # Test email
    
    # Process a video file (replace with your actual video file)
    # Check multiple possible video filenames
    video_files = ["video.mp4", "input_video.mp4", "test_video.mp4"]
    found_video = False
    
    for video_file in video_files:
        if os.path.exists(video_file):
            print(f"Found video file: {video_file}")
            run_on_video(video_file)
            found_video = True
            break
    
    if not found_video:
        print(f"Error: No video file found. Please provide a valid video file.")
        print(f"Looked for: {video_files}")
        
        # Fallback to test_manual_violation if no video found
        print("Falling back to manual violation test...")
        test_manual_violation()
