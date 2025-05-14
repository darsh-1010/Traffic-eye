import streamlit as st
import tempfile
import os
import cv2
from PIL import Image
import numpy as np
import csv

# Import your main detection functions
from try_final import process_video, process_frame

st.set_page_config(page_title="Traffic Violation Detection", layout="wide")
st.title("🚦 Traffic Violation Detection System")
st.markdown("""
Analyze **video or image** for:
- Helmet violations
- Multiple riders
- PUC expiry detection
- License plate logging and email alerts
""")

# Sidebar for input selection
input_type = st.sidebar.radio("Select Input Type", ["📷 Image", "🎥 Video", "🔎 CSV Search"])


# After the imports at the top of the file, add this function
def read_vehicle_registry():
    """Read the vehicle registry directly from CSV to ensure fresh data"""
    registry = {}
    try:
        with open('vehicle_registry.csv', 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                plate = row['Plate_Number'].strip().upper()
                registry[plate] = dict(row)
        st.sidebar.success(f"✅ Loaded {len(registry)} vehicles from registry")
    except Exception as e:
        st.sidebar.error(f"❌ Failed to load vehicle registry: {e}")
    return registry

# Add a registry refresh button in the sidebar
st.sidebar.write("---")
if st.sidebar.button("🔄 Refresh Vehicle Registry"):
    fresh_registry = read_vehicle_registry()
    # This won't update try_final.vehicle_registry, but can be used locally
    st.sidebar.write(f"Registry contains {len(fresh_registry)} vehicles")

# ------------------------------------
# 📷 IMAGE UPLOAD AND PROCESSING
# ------------------------------------
if input_type == "📷 Image":
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        image = Image.open(uploaded_image)
        image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.info("🔍 Processing image for traffic violations...")

        # Create video summary to collect detection results
        video_summary = {
            'plates': [], 'helmets': [], 'without_helmets': [],
            'vehicles': [], 'violations': []
        }

        # Process the image and get results
        processed_frame, vehicles = process_frame(
            image_np, frame_count=1, fps=30, total_frames=1, video_summary=video_summary
        )

        # Display the processed image
        st.image(processed_frame, channels="BGR", caption="🔎 Detection Result")
        
        # Check for violations in detected vehicles
        has_violations = False
        
        # Get a fresh copy of the registry for our display
        fresh_registry = read_vehicle_registry()
        
        # Traffic violations (helmet, multiple riders)
        for vehicle in vehicles:
            if vehicle['without_helmet'] > 0 or vehicle['total_riders'] > 2:
                has_violations = True
                # Add to violations list if not already there
                violation_exists = False
                for v in video_summary['violations']:
                    if v.get('vehicle_id') == vehicle['vehicle_id']:
                        violation_exists = True
                        break
                
                if not violation_exists:
                    video_summary['violations'].append({
                        'frame': 1,
                        'vehicle_id': vehicle['vehicle_id'],
                        'plate_number': vehicle['plate_number'],
                        'total_riders': vehicle['total_riders'],
                        'without_helmet': vehicle['without_helmet'],
                        'timestamp': 0
                    })
        
        # PUC Expiry violations
        from try_final import vehicle_registry, parse_puc_date
        from datetime import datetime
        
        plates_with_puc_violations = set()
        # Check all detected plates for PUC expiry
        for plate in video_summary['plates']:
            plate_text = plate['text']
            if plate_text != "Unknown" and (plate_text in vehicle_registry or plate_text in fresh_registry):
                # Check both registries
                info = vehicle_registry.get(plate_text, fresh_registry.get(plate_text, {}))
                try:
                    puc_date = parse_puc_date(info['PUC_Expiry_Date'], plate_text)
                    if puc_date and puc_date < datetime.now():
                        plates_with_puc_violations.add(plate_text)
                        has_violations = True
                except Exception as e:
                    st.warning(f"Could not check PUC expiry for {plate_text}: {e}")
        
        if has_violations or video_summary['violations'] or plates_with_puc_violations:
            st.error("🚨 Violations Detected!")
            
            # Show violation counts
            helmet_violations = sum(1 for v in vehicles if v['without_helmet'] > 0)
            rider_violations = sum(1 for v in vehicles if v['total_riders'] > 2)
            
            if helmet_violations > 0:
                st.warning(f"🪖 {helmet_violations} vehicle(s) with riders not wearing helmets")
            
            if rider_violations > 0:
                st.warning(f"👥 {rider_violations} vehicle(s) with more than 2 riders")
                
            if plates_with_puc_violations:
                st.warning(f"📃 {len(plates_with_puc_violations)} vehicle(s) with expired PUC certificate")
                st.warning("PUC Expired for plates: " + ", ".join(plates_with_puc_violations))
        else:
            st.success("✅ No violations found.")
        
        # Show detected plates
        if video_summary['plates']:
            with st.expander("🪪 Detected License Plates"):
                for plate in video_summary['plates']:
                    if plate['text'] != "Unknown":
                        if plate['text'] in plates_with_puc_violations:
                            st.write(f"• {plate['text']} (⚠️ PUC Expired)")
                        else:
                            st.write(f"• {plate['text']}")

        # Show detailed violations
        if video_summary['violations'] or plates_with_puc_violations:
            with st.expander("📄 View Detailed Violations"):
                # Traffic violations
                if video_summary['violations']:
                    st.subheader("Traffic Violations")
                    for v in video_summary['violations']:
                        violations_text = []
                        if v['without_helmet'] > 0:
                            violations_text.append(f"No Helmet ({v['without_helmet']} rider(s))")
                        if v['total_riders'] > 2:
                            violations_text.append(f"Multiple Riders ({v['total_riders']} total)")
                        
                        st.write(f"🔴 Vehicle {v['vehicle_id']} with plate {v['plate_number']}")
                        st.write(f"   Violations: {', '.join(violations_text)}")
                        st.write("---")
                
                # PUC violations
                if plates_with_puc_violations:
                    st.subheader("PUC Expiry Violations")
                    
                    # Use the fresh registry for display
                    registry_to_use = fresh_registry if fresh_registry else vehicle_registry
                    
                    for plate in plates_with_puc_violations:
                        st.write(f"🔴 Vehicle with plate: {plate}")
                        
                        if plate in registry_to_use:
                            info = registry_to_use[plate]
                            
                            # Debug output to see the actual dictionary format
                            st.write("Registry info for this plate:")
                            st.json(info)
                            
                            # Try different ways to access the data
                            owner_name = info.get('Name', 'Not found')
                            if owner_name == 'Not found' or owner_name == 'N/A':
                                # Try alternate capitalization
                                owner_name = info.get('name', info.get('NAME', 'Unknown'))
                            
                            phone = info.get('Phone', 'Not found')
                            if phone == 'Not found' or phone == 'N/A':
                                # Try alternate capitalization
                                phone = info.get('phone', info.get('PHONE', 'Unknown'))
                            
                            puc_date = info.get('PUC_Expiry_Date', 'Not found')
                            if puc_date == 'Not found':
                                # Try alternate formats
                                puc_date = info.get('puc_expiry_date', info.get('Puc_Expiry_Date', 'Unknown'))
                            
                            st.write(f"   Owner: {owner_name}")
                            st.write(f"   Phone: {phone}")
                            st.write(f"   PUC Expiry: {puc_date}")
                        else:
                            st.write("   ❌ Plate not found in registry!")
                            st.write(f"   Available plates: {list(registry_to_use.keys())}")
                        
                        st.write("---")

# ------------------------------------
# 🎥 VIDEO UPLOAD AND PROCESSING
# ------------------------------------
elif input_type == "🎥 Video":
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])

    if uploaded_video:
        st.video(uploaded_video)

        # Save uploaded video to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_input:
            tmp_input.write(uploaded_video.read())
            input_video_path = tmp_input.name

        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        output_video_path = temp_output.name

        st.info("⏳ Processing video. This may take some time...")
        video_summary = process_video(input_video_path, output_video_path, process_every_nth_frame=10)
        st.success("✅ Video processed successfully!")

        # Ensure file is written and flushed before reading
        temp_output.close()
        import time
        time.sleep(0.5)



        # Summary and violation display code (unchanged)
        helmet_violations = sum(1 for v in video_summary['violations'] if 'without_helmet' in v and v['without_helmet'] > 0)
        multiple_rider_violations = sum(1 for v in video_summary['violations'] if 'total_riders' in v and v['total_riders'] > 2)
        
        plates_with_puc_violations = set()
        from try_final import vehicle_registry, parse_puc_date
        from datetime import datetime

        fresh_registry = read_vehicle_registry()
        registry_to_use = fresh_registry if fresh_registry else vehicle_registry

        for plate in set([p['text'] for p in video_summary['plates'] if p['text'] != 'Unknown']):
            if plate in registry_to_use:
                try:
                    puc_date = parse_puc_date(registry_to_use[plate]['PUC_Expiry_Date'], plate)
                    if puc_date and puc_date < datetime.now():
                        plates_with_puc_violations.add(plate)
                except Exception as e:
                    st.warning(f"Could not check PUC expiry for {plate}: {e}")

        total_violations = len(video_summary['violations']) + len(plates_with_puc_violations)

        with st.expander("📋 Summary of Violations"):
            st.write(f"🔢 Total Violations: {total_violations}")
            st.write(f"🪪 Unique Plates Detected: {len(set([p['text'] for p in video_summary['plates'] if p['text'] != 'Unknown']))}")
            st.write(f"🪖 Riders with Helmet: {len(video_summary['helmets'])}")
            st.write(f"🚫 Riders without Helmet: {len(video_summary['without_helmets'])}")
            
            if helmet_violations > 0:
                st.write(f"🚨 Helmet Violations: {helmet_violations}")
            if multiple_rider_violations > 0:
                st.write(f"🚨 Multiple Rider Violations: {multiple_rider_violations}")
            if plates_with_puc_violations:
                st.write(f"🚨 PUC Expiry Violations: {len(plates_with_puc_violations)}")
                st.write("PUC Expired for plates: " + ", ".join(plates_with_puc_violations))

        with st.expander("📄 Detailed Violation Logs"):
            st.subheader("Traffic Violations")
            for v in video_summary['violations']:
                st.json(v)

            st.subheader("PUC Violations")
            if plates_with_puc_violations:
                for plate in plates_with_puc_violations:
                    st.write(f"🔴 Vehicle with plate: {plate}")
                    if plate in registry_to_use:
                        info = registry_to_use[plate]
                        st.json(info)
                    else:
                        st.write("   ❌ Plate not found in registry!")
            else:
                st.write("No PUC expiry violations detected.")

        with st.expander("📊 All Detected Plates"):
            for p in video_summary['plates']:
                st.write(f"• {p['text']} (Frame: {p['frame']})")

# ------------------------------------
# 🔎 CSV REGISTRY SEARCH AND DISPLAY SECTION
# ------------------------------------

elif input_type == "🔎 CSV Search":

    st.header("🔎 Search Vehicle Registry")
    import pandas as pd

    @st.cache_data
    def load_registry_df():
        try:
            df = pd.read_csv('vehicle_registry.csv')
            return df
        except Exception as e:
            st.error(f"❌ Error loading CSV: {e}")
            return pd.DataFrame()

    registry_df = load_registry_df()

    search_input = st.text_input("🔍 Enter Plate Number, Owner Name or Phone to search:")

    if search_input:
        filtered_df = registry_df[
            registry_df.apply(lambda row: row.astype(str).str.contains(search_input, case=False).any(), axis=1)
        ]
        st.success(f"✅ Found {len(filtered_df)} matching result(s).")
    else:
        filtered_df = registry_df.copy()

    if not filtered_df.empty:
        st.dataframe(filtered_df, use_container_width=True)
    else:
        st.info("No matching results found.")
