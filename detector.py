import cv2
import time
import numpy as np
from picamera2 import Picamera2
import mediapipe as mp
import os
import globals
from constants import *

class Detector:
    def __init__(self, resolution=(960, 800), fps=30):
        self.resolution = resolution
        self.fps = fps

        self.camera = Picamera2()
        video_config = self.camera.create_video_configuration(
            main={"size": resolution, "format": "BGR888"})
        
        self.camera.configure(video_config)
        self.camera.start()

        self.empty_eye_frame = np.zeros((const_single_eye_window_dim, const_single_eye_window_dim*2), dtype=np.uint8)
        self.empty_single_eye_frame = np.zeros((const_single_eye_window_dim, const_single_eye_window_dim), dtype=np.uint8)
        
        self.face_cascade = None
        self.eye_cascade = None
        self.yolo_session = None
        self.input_name = None
        self.face_mesh = None
        self.mp_face_mesh = None
        
        self.prev_time = time.time()
        self.fps = 0
        

    def preprocess(self, frame):
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return gray


    def preview(self, frame, window_name):     
        cv2.imshow(window_name, frame)


    def detect_face_haarcascade(self, gray_frame):
        faces = self.face_cascade.detectMultiScale(
            gray_frame, scaleFactor=1.3, minNeighbors=4, minSize=const_minimum_face_size
        )
        if const_draw_face_rectangles:
            for (x, y, w, h) in faces:
                cv2.rectangle(gray_frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
        return faces


    def detect_eyes_haarcascade(self, gray_frame, faces):
        """
        Purpose:
            Wykrywa oczy w obrębie twarzy przy użyciu klasyfikatora Haarcascade.
            Zwraca przeskalowane obrazy lewego i prawego oka.

        Parameters:
            gray_frame (np.ndarray): Obraz w odcieniach szarości.
            faces (list[tuple]): Lista wykrytych twarzy (x, y, w, h).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Obrazy lewego i prawego oka w rozmiarze const_single_eye_window_dim.
        """
        left_eye_frame = np.zeros_like(self.empty_single_eye_frame)
        right_eye_frame = np.zeros_like(self.empty_single_eye_frame)

        for (x, y, w, h) in faces:
            roi_gray = gray_frame[y:y + h, x:x + w]
            eyes = self.eye_cascade.detectMultiScale(
                roi_gray, scaleFactor=1.15, minNeighbors=3,
                minSize=(int(h * const_min_eye_proportions), int(w * const_min_eye_proportions)),
                maxSize=(int(h * const_max_eye_proportions), int(w * const_max_eye_proportions))
                                                                                                )
            eyes = sorted(eyes, key=lambda x: x[0])

            if len(eyes) >= 1:
                ex, ey, ew, eh = eyes[0]
                eye_roi = roi_gray[ey:ey + eh, ex:ex + ew]
                if eye_roi.size == 0:
                    continue  
                cv2.resize(eye_roi, (const_single_eye_window_dim, const_single_eye_window_dim),
                             dst=left_eye_frame, interpolation=cv2.INTER_LINEAR)

            if len(eyes) >= 2:
                ex, ey, ew, eh = eyes[1]
                eye_roi = roi_gray[ey:ey + eh, ex:ex + ew]
                if eye_roi.size == 0:
                    continue  
                cv2.resize(eye_roi, (const_single_eye_window_dim, const_single_eye_window_dim),
                             dst=right_eye_frame, interpolation=cv2.INTER_LINEAR)

        return left_eye_frame, right_eye_frame

    def detect_eyes_mediapipe(self, gray_frame):
        """
        Purpose:
            Wykrywa oczy przy użyciu MediaPipe FaceMesh.
            Skaluje oczy do wymiarów const_single_eye_window_dim.

        Parameters:
            gray_frame (np.ndarray): Obraz w odcieniach szarości.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Obrazy lewego i prawego oka.
        """
        rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)
        results = self.face_mesh.process(rgb_frame)
    
        left_eye_frame = self.empty_single_eye_frame
        right_eye_frame = self.empty_single_eye_frame

        if not results.multi_face_landmarks:
            return left_eye_frame, right_eye_frame

        h, w = gray_frame.shape

        left_eye_idx = [const_mp_facemesh_left_eye_lower_left_corner,
                         const_mp_facemesh_left_eye_lower_right_corner,
                         const_mp_facemesh_left_eye_upper_left_corner,
                         const_mp_facemesh_left_eye_upper_right_corner]
        
        right_eye_idx = [const_mp_facemesh_right_eye_lower_left_corner,
                         const_mp_facemesh_right_eye_lower_right_corner,
                         const_mp_facemesh_right_eye_upper_left_corner,
                         const_mp_facemesh_right_eye_upper_right_corner,]

        for face_landmarks in results.multi_face_landmarks:
            left_x = [int(face_landmarks.landmark[i].x * w) for i in left_eye_idx]
            left_y = [int(face_landmarks.landmark[i].y * h) for i in left_eye_idx]
            right_x = [int(face_landmarks.landmark[i].x * w) for i in right_eye_idx]
            right_y = [int(face_landmarks.landmark[i].y * h) for i in right_eye_idx]

            lx_min, lx_max = max(min(left_x), 0), min(max(left_x), w)
            ly_min, ly_max = max(min(left_y), 0), min(max(left_y), h)
            rx_min, rx_max = max(min(right_x), 0), min(max(right_x), w)
            ry_min, ry_max = max(min(right_y), 0), min(max(right_y), h)
    
            left_eye = gray_frame[ly_min:ly_max, lx_min:lx_max]
            right_eye = gray_frame[ry_min:ry_max, rx_min:rx_max]

            if left_eye.size != 0:
                left_eye_frame = cv2.resize(left_eye, (const_single_eye_window_dim, const_single_eye_window_dim))
            else:
                left_eye_frame = self.empty_single_eye_frame

            if right_eye.size != 0:
                right_eye_frame = cv2.resize(right_eye, (const_single_eye_window_dim, const_single_eye_window_dim))
            else:
                right_eye_frame = self.empty_single_eye_frame

        return left_eye_frame, right_eye_frame

 
    def stitch_multiple_eyes_horizontal(self, left_eye, right_eye):
        if left_eye is None or right_eye is None:
            return self.empty_eye_frame
        return cv2.hconcat([left_eye, right_eye])


    def find_darkest_pixels_and_glint(self, eye, const_ignore_bounds_x, const_ignore_bounds_y, const_image_skip_size, const_search_area, const_internal_skip_size):
        """
        Purpose:
            Analizuje obszar oka i znajduje punkt najciemniejszy i najbardziej jednolity obszar pikseli (zwykle na źrenicy)
            oraz najjaśniejszy punkt w jego pobliżu (glint).

        Parameters:
            eye (np.ndarray): Obraz oka.
            const_ignore_bounds_x/y (int): Margines ignorowany przy krawędziach.
            const_image_skip_size (int): Krok iteracji w pikselach.
            const_search_area (int): Wielkość okna do analizy pikseli.
            const_internal_skip_size (int): Krok iteracji wewnątrz okna.

        Returns:
            Tuple[Tuple[int,int], int, Tuple[int,int], int]: 
                - Punkt najciemniejszy, jego wartość pikseli,
                - Punkt najjaśniejszy, jego wartość pikseli.

        Side Effects:
            - Brak.
        """       
        min_sum = float('inf')
        darkest_point = None
        darkest_pixel_value = None
        
        max_value = -float('inf')
        brightest_point = None
        brightest_pixel_value = None
    
        if eye is None:
           return None, None, None, None
            
        for y in range(const_ignore_bounds_y, eye.shape[0] - const_ignore_bounds_y, const_image_skip_size):
            for x in range(const_ignore_bounds_x, eye.shape[1] - const_ignore_bounds_x, const_image_skip_size):

                current_sum = 0
                num_pixels = 0               

                for dy in range(0, const_search_area, const_internal_skip_size):
                    y_pos = y + dy
                    if y_pos >= eye.shape[0]:
                        break

                    for dx in range(0, const_search_area, const_internal_skip_size):
                        x_pos = x + dx
                        if x_pos >= eye.shape[1]:
                            break

                        pixel_value = int(eye[y_pos][x_pos])
                        current_sum += pixel_value
                        num_pixels += 1              
         
                if current_sum < min_sum and num_pixels > 0:
                    min_sum = current_sum
                    darkest_point = (x + const_search_area // 2, y + const_search_area // 2)
                    darkest_pixel_value = int(eye[darkest_point[1], darkest_point[0]])
                
                if pixel_value > max_value and pixel_value >= const_minimum_glint_threshold and x_pos in range(darkest_point[0] - const_glint_vector_length_limit_x, darkest_point[0] + const_glint_vector_length_limit_x) and y_pos in range (darkest_point[1] - const_glint_vector_length_limit_y, darkest_point[1] + const_glint_vector_length_limit_y):
                    max_value = pixel_value
                    brightest_point = (x_pos, y_pos)
                    brightest_pixel_value = pixel_value

        return darkest_point, darkest_pixel_value, brightest_point, brightest_pixel_value
    

    def filter_contours_by_area_and_return_largest(self, contours, const_pixel_threshold, const_ratio_threshold):
        """
        Purpose:
            Filtruje kontury według minimalnej powierzchni i stosunku długości do szerokości,
            zwraca największy kontur spełniający kryteria.

        Parameters:
            contours (list[np.ndarray]): Lista konturów do sprawdzenia.
            const_pixel_threshold (float): Minimalna powierzchnia konturu.
            const_ratio_threshold (float): Maksymalny stosunek długości do szerokości.

        Returns:
            list[np.ndarray]: Lista zawierająca największy zaakceptowany kontur lub pustą listę.
        """
        max_area = 0
        largest_contour = None

        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= const_pixel_threshold:
                x, y, w, h = cv2.boundingRect(contour)
                length_to_width_ratio = max(w / h, h / w)
                if length_to_width_ratio <= const_ratio_threshold:
                    if area > max_area:
                        max_area = area
                        largest_contour = contour

        return [largest_contour] if largest_contour is not None else []
   

    def mask_outside_square(self, thresholded_eyes, darkest_point, const_halfmask_size):
        if darkest_point is None:
            return self.empty_eye_frame
        mask = np.zeros_like(thresholded_eyes)
        
        top_left_x = max(0, darkest_point[0] - const_halfmask_size)
        top_left_y = max(0, darkest_point[1] - const_halfmask_size)
        bottom_right_x = min(thresholded_eyes.shape[1], darkest_point[0] + const_halfmask_size)
        bottom_right_y = min(thresholded_eyes.shape[0], darkest_point[1] + const_halfmask_size)
        
        mask[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 255
        
        return cv2.bitwise_and(thresholded_eyes, mask)
    
    
    def detect_pupil(self, eyes, darkest_point, darkest_pixel_value):
        """
        Purpose:
            Progowanie obrazu oka w celu znalezienia źrenicy.
            Zastosowanie maski wokół wykrytego obszaru.

        Parameters:
            eyes (np.ndarray): Obraz oka.
            darkest_point (Tuple[int,int]): Współrzędne źrenicy.
            darkest_pixel_value (int): Wartość najciemniejszego piksela.

        Returns:
            np.ndarray: Obraz binarny źrenicy.
        """
        if eyes is None or darkest_point is None or darkest_pixel_value is None:
            return self.empty_eye_frame
               
        threshold = int(darkest_pixel_value + const_added_threshold)
        _, thresholded_eyes = cv2.threshold(eyes, threshold, 255, cv2.THRESH_BINARY_INV)
               
        masked_eyes = self.mask_outside_square(thresholded_eyes, darkest_point, const_halfmask_size)
                 
        return masked_eyes
   
   
    def fit_elipse_to_pupil(self, pupil, eye, const_pixel_threshold, const_ratio_threshold, const_min_contour_points, brightest_point):
        """
        Purpose:
            Dopasowuje elipsę do konturu źrenicy, wyznacza środek i stosunek osi elipsy reprezentującą źrenicę.
            Określa, czy oko jest zamknięte i rysuje elipsę oraz środek elipsy na obrazie.

        Parameters:
            pupil (np.ndarray): Obraz binarny źrenicy.
            eye (np.ndarray): Oryginalny obraz oka.
            const_pixel_threshold (float): Minimalna powierzchnia konturu do akceptacji.
            const_ratio_threshold (float): Maksymalny stosunek długości do szerokości konturu.
            const_min_contour_points (int): Minimalna liczba punktów konturu do dopasowania elipsy.
            brightest_point (Tuple[int,int]): Współrzędne najjaśniejszego punktu (glint).

        Returns:
            Tuple[np.ndarray, int, int, float, bool]: 
                - Obraz oka z elipsą,
                - współrzędne środka źrenicy (x, y),
                - stosunek osi elipsy,
                - flaga zamkniętego oka.
        """
        if pupil is None:
            return self.empty_eye_frame, 0, 0, 1.0, True  
        if eye is None:
            return self.empty_eye_frame, 0, 0, 1.0, True
    
        kernel = np.ones((const_kernel_size, const_kernel_size), np.uint8)
        dilated_image = cv2.dilate(pupil, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        reduced_contours = self.filter_contours_by_area_and_return_largest(contours, const_pixel_threshold, const_ratio_threshold)
        
        
        eye_with_ellipse = eye.copy()
        pupil_center_x, pupil_center_y = 0, 0
        ellipse_ratio = 1.0
        eye_closed = False
    
        if len(reduced_contours) > 0 and len(reduced_contours[0]) > const_min_contour_points:
            ellipse = cv2.fitEllipse(reduced_contours[0])
            (center, axes, angle) = ellipse
            pupil_center_x, pupil_center_y = map(int, center)
            eye_with_ellipse = cv2.ellipse(eye_with_ellipse, ellipse, (0, 255, 0), 2)
            eye_with_ellipse = cv2.circle(eye_with_ellipse, (pupil_center_x, pupil_center_y), 3, (255, 255, 0), -1)
            major_axis, minor_axis = max(axes), min(axes)
            ellipse_ratio = minor_axis / major_axis
                        
            
            if ellipse_ratio < const_elipse_ratio_threshold and len(reduced_contours[0]) > const_blink_contour_points_threshold and brightest_point is None:
                eye_closed = True
            if eye_closed is True:
                eye_with_ellipse = self.empty_single_eye_frame

        return eye_with_ellipse, pupil_center_x, pupil_center_y, ellipse_ratio, eye_closed

 
    def draw_info(self, eye_with_ellipse, pupil_center_x, pupil_center_y, brightest_point, eye_closed):
        if const_draw_info is True and eye_with_ellipse is not None:
            if eye_closed is True: 
                cv2.putText(eye_with_ellipse, "Closed", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            elif eye_closed is False:
                cv2.putText(eye_with_ellipse, "Open", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
      
            if pupil_center_x != 0 and pupil_center_y != 0 and brightest_point is not None and eye_closed is False:
                brightest_point_x, brightest_point_y = brightest_point[0], brightest_point[1]
                cv2.putText(eye_with_ellipse, f"[{pupil_center_x}, {pupil_center_y}]", (pupil_center_x - 2, pupil_center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2) 
                cv2.circle(eye_with_ellipse, (brightest_point_x, brightest_point_y), 2, (255, 0, 0), 1)
                cv2.putText(eye_with_ellipse, f"[{brightest_point_x}, {brightest_point_y}]", (brightest_point_x - 2, brightest_point_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
                cv2.line(eye_with_ellipse, (pupil_center_x, pupil_center_y), (brightest_point_x, brightest_point_y), (255, 0, 0), 2)
            if const_draw_even_more_info:
                cv2.rectangle(eye_with_ellipse,
                    (const_ignore_bounds_x + const_search_area // 2, const_ignore_bounds_y + const_search_area // 2),
                    (eye_with_ellipse.shape[1] - const_ignore_bounds_x - const_search_area // 2,
                    eye_with_ellipse.shape[0] - const_ignore_bounds_y - const_search_area // 2),
                    (0, 255, 0), 1)
               
    
    def create_vector(self, pupil_center, brightest_point):
        """
        Purpose:
            Tworzy wektor cech oka na podstawie środka źrenicy, a punktem odbicia światła (glint).
            Oblicza długość i kąt wektora w układzie współrzędnych związanym z obrazem pojedynczego oka.

        Parameters:
            pupil_center (Tuple[int,int]): Współrzędne środka źrenicy.
            brightest_point (Tuple[int,int]): Współrzędne punktu odbicia światła.

        Returns:
            np.ndarray: Wektor cech [długość, kąt]
        """
        vector_length = None
        vector_angle = None
        eye_feature_vector = None
        
        if pupil_center is not None and brightest_point is not None:
            dx = pupil_center[0] - brightest_point[0]
            dy = pupil_center[1] - brightest_point[1]
            vector_length = np.sqrt(dx * dx + dy * dy)
            vector_angle = np.arctan2(dy, dx)
            
            eye_feature_vector = np.array([vector_length, vector_angle])
            
        return eye_feature_vector


    def choose_eye_detection_algorithm(self):
        while True:
            print("\n=== Wybierz algorytm detekcji oczu ===")
            print("1. Haarcascade")
            print("2. MediaPipe FaceMesh")
            print("--------------------------------------")
            choice = input("Wybierz [1/2]: ").strip()
            match choice:
                
                case '1':
                    print("Ładowanie klasyfikatorów Haarcascade ...")
                    self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                    self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
                    if self.face_cascade.empty() or self.eye_cascade.empty():
                        raise RuntimeError("Nie udało się załadować klasyfikatorów Haarcascade")
                    print("Klasyfikatory Haarcascade załadowane pomyślnie")

                case '2':
                    print("Inicjalizacja MediaPipe FaceMesh ...")                    
                    self.mp_face_mesh = mp.solutions.face_mesh
                    self.face_mesh = self.mp_face_mesh.FaceMesh(
                        max_num_faces=2,
                        refine_landmarks=True,
                        min_detection_confidence=0.7,
                        min_tracking_confidence=0.7,
                    )
                    print("MediaPipe FaceMesh zainicjalizowany")

                case _:
                    print("Nieprawidłowy wybór. Spróbuj ponownie.")
                    continue

            return choice


    def process_frame(self, gray_frame, algorithm_choice):
        """
        Purpose:
            Przetwarza pojedynczą klatkę obrazu w celu wykrycia twarzy i oczu.
            Wyznacza pozycję środka źrenicy i glint, dopasowuje elipsę do źrenicy,
            generuje wektor cech oka i aktualizuje globalne zmienne ostatnich wektorów.

        Parameters:
            gray_frame (np.ndarray): Obraz w odcieniach szarości.
            algorithm_choice (str): Wybrany algorytm detekcji oczu ('1'=Haarcascade, '2'=YOLO, '3'=MediaPipe).

        Returns:
            Tuple[list, np.ndarray, np.ndarray, bool, bool]:
                - faces: lista wykrytych twarzy,
                - pupils: obraz pupili złączony poziomo,
                - eyes_with_ellipse: obraz oczu z narysowanymi elipsami,
                - left_eye_closed: flaga zamknięcia lewego oka,
                - right_eye_closed: flaga zamknięcia prawego oka.
        """
        match algorithm_choice:           
            case '1':
                faces = self.detect_face_haarcascade(gray_frame)
                left_eye, right_eye = self.detect_eyes_haarcascade(gray_frame, faces)                               

            case '2':
                left_eye, right_eye = self.detect_eyes_mediapipe(gray_frame)
                faces = []                

            case _:
                return [], self.empty_eye_frame, self.empty_eye_frame, None, None
            
        left_eye_darkest_point, left_eye_darkest_pixel_value, left_eye_brightest_point, left_eye_brightest_pixel_value = self.find_darkest_pixels_and_glint(left_eye, const_ignore_bounds_x, const_ignore_bounds_y, const_image_skip_size, const_search_area, const_internal_skip_size)
        right_eye_darkest_point, right_eye_darkest_pixel_value, right_eye_brightest_point, right_eye_brightest_pixel_value = self.find_darkest_pixels_and_glint(right_eye,  const_ignore_bounds_x, const_ignore_bounds_y, const_image_skip_size, const_search_area, const_internal_skip_size)        
                
        left_pupil = self.detect_pupil(left_eye, left_eye_darkest_point, left_eye_darkest_pixel_value)
        right_pupil = self.detect_pupil(right_eye, right_eye_darkest_point, right_eye_darkest_pixel_value)
                
        left_eye_with_ellipse, left_pupil_center_x, left_pupil_center_y, left_ellipse_ratio, left_eye_closed = self.fit_elipse_to_pupil(left_pupil, left_eye, const_pixel_threshold, const_ratio_threshold, const_min_contour_points, left_eye_brightest_point)
        right_eye_with_ellipse, right_pupil_center_x, right_pupil_center_y,right_ellipse_ratio, right_eye_closed = self.fit_elipse_to_pupil(right_pupil, right_eye, const_pixel_threshold, const_ratio_threshold, const_min_contour_points, right_eye_brightest_point)
                
        self.draw_info(left_eye_with_ellipse, left_pupil_center_x, left_pupil_center_y, left_eye_brightest_point, left_eye_closed)
        self.draw_info(right_eye_with_ellipse, right_pupil_center_x, right_pupil_center_y, right_eye_brightest_point, right_eye_closed)
                
        left_pupil_center = [left_pupil_center_x, left_pupil_center_y]
        right_pupil_center = [right_pupil_center_x, right_pupil_center_y]
        
        left_pupil_glint_vector = self.create_vector(left_pupil_center, left_eye_brightest_point)
        right_pupil_glint_vector = self.create_vector(right_pupil_center, right_eye_brightest_point)
        
        
        globals.last_left_vector = left_pupil_glint_vector
        globals.last_right_vector = right_pupil_glint_vector
                              
        pupils = self.stitch_multiple_eyes_horizontal(left_pupil, right_pupil)
        eyes_with_ellipse = self.stitch_multiple_eyes_horizontal(left_eye_with_ellipse, right_eye_with_ellipse)
            
        return faces, pupils, eyes_with_ellipse, left_eye_closed, right_eye_closed    
   
   
    def update_fps(self):
        current_time = time.time()
        delta = current_time - self.prev_time
        self.prev_time = current_time

        if delta > 0:
            self.fps = 1.0 / delta
            
        return round(self.fps, 1)
    
   
    def get_features(self, algorithm_choice):        
        frame = self.camera.capture_array()
        gray_frame = self.preprocess(frame)
        faces, pupils, eyes_with_ellipse, left_eye_closed, right_eye_closed = self.process_frame(gray_frame, algorithm_choice)
        
        return gray_frame, faces, pupils, eyes_with_ellipse, left_eye_closed, right_eye_closed
  
  
    def full_preview(self, gray_frame, faces, pupils, eyes_with_ellipse):      
        fps_value = self.update_fps()        
        cv2.putText(gray_frame, f"FPS: {fps_value}",(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        self.preview(gray_frame, "Live Feed")
        self.preview(pupils, "Pupils")
        self.preview(eyes_with_ellipse, "Eyes")
        
            
    def stop(self):
        self.camera.stop()
        cv2.destroyAllWindows()
        cv2.waitKey(100)
        print("Koniec nagrania")
