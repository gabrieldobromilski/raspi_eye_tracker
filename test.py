import sys

def check_numpy():
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__} is installed.")
        # quick test: simple array operation
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        if (a + b == [5, 7, 9]).all():
            print("   NumPy basic operations OK.")
        else:
            print("   ⚠️ NumPy operation test failed.")
    except Exception as e:
        print("❌ NumPy check failed:", e)


def check_opencv():
    try:
        import cv2
        print(f"✅ OpenCV {cv2.__version__} is installed.")
        # quick test: create a blank image
        import numpy as np
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.line(img, (0, 0), (99, 99), (255, 0, 0), 2)
        print("   OpenCV basic image ops OK.")
    except Exception as e:
        print("❌ OpenCV check failed:", e)


def check_picamera2():
    try:
        from picamera2 import Picamera2
        picam2 = Picamera2()
        print("✅ Picamera2 is installed and importable.")
        # Try opening the camera
        picam2.start()
        print("   Picamera2 camera started successfully.")
        picam2.stop()
        print("   Picamera2 camera stopped successfully.")
    except ImportError:
        print("❌ Picamera2 is not installed.")
    except Exception as e:
        print("⚠️ Picamera2 import succeeded but camera test failed:", e)


def check_ultralytics():
    try:
        from ultralytics import YOLO
        print("✅ Ultralytics YOLO is installed.")

        # Quick test: load a small pretrained model and run a dummy inference
        model = YOLO("yolov8n.pt")  # nano model, small and fast
        print("   YOLO model loaded successfully.")

        # Dummy inference on an empty image (640x640)
        import numpy as np
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        results = model(dummy_img)
        print("   YOLO dummy inference ran successfully.")
    except ImportError:
        print("❌ Ultralytics YOLO is not installed.")
    except Exception as e:
        print("⚠️ Ultralytics YOLO import or inference failed:", e)


def check_mediapipe():
    try:
        import mediapipe as mp, numpy as np, cv2
        mp.solutions.face_mesh.FaceMesh(static_image_mode=True).process(
            cv2.cvtColor(np.zeros((480,640,3), np.uint8), cv2.COLOR_BGR2RGB)
        ).multi_face_landmarks  # dummy inference
        print("✅ Mediapipe loads and runs.")
    except ImportError:
        print("❌ Mediapipe not installed.")
    except Exception as e:
        print("⚠️ Mediapipe test failed:", e)




if __name__ == "__main__":
    print("🔍 Checking Python environment...\n")
    check_numpy()
    print("\n")
    check_opencv()
    print("\n")
    check_picamera2()
    print("\n")
    check_ultralytics()
    print("\n")
    check_mediapipe()
    print("\n✅ Done checking.\n")

