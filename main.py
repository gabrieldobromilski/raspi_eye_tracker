from detector import Detector
from gaze_mapper import GazeMapper
from gaze_renderer import GazeRenderer
from calibrator import CalibrationCollector
from menu import Menu
import globals
import cv2
import time
import warnings
import os

if __name__ == "__main__":
    try:
        warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")
        det = Detector()
        menu = Menu()
        os.system("clear")
        mode = menu.mode_choice()
        os.system("clear")
        calibration_file_status_left = CalibrationCollector.does_calibration_file_exist("calibration_data_left.npz")
        calibration_file_status_right = CalibrationCollector.does_calibration_file_exist("calibration_data_right.npz")
        
        if (mode == "2" or mode == "4") and (calibration_file_status_left is False or calibration_file_status_right is False):
            print("Nie wykryto pliku jednego lub więcej calibration_data.npz następuje kalibracja")
            mode = "1"
            
        algorithm_choice = det.choose_eye_detection_algorithm()        
        os.system("clear")
        
        match mode:
            case "1":
                print("\nWybrano: NOWA KALIBRACJA\n")

                time.sleep(1)
                print("Kalibracja za 3...")
                time.sleep(1)
                print("Kalibracja za 2...")
                time.sleep(1)
                print("Kalibracja za 1...")
                time.sleep(1)
                os.system("clear")
                calib = CalibrationCollector()                
                while True:
                    gray_frame, faces, pupils, eyes_with_ellipse, is_left_eye_closed, is_right_eye_closed = det.get_features(algorithm_choice)
                    calib.run_calibration()

                    if calib.is_calibration_ready():
                        print("Kalibracja zakończona")
                        calib.save_calibration()
                        calib.pygame_quit()
                        break                    
                
            case "2":
                print("\nWybrano: UŻYCIE ISTNIEJĄCEGO PLIKU\n")
                print("Ładowanie calibration_data.npz...")
                time.sleep(1)
                left_mapper = GazeMapper("left")
                left_mapper.load_calibration("calibration_data_left.npz")
                
                right_mapper = GazeMapper("right")
                right_mapper.load_calibration("calibration_data_right.npz")
                
                gaze_point_list = []
                
                renderer = GazeRenderer()                
                os.system("clear")
                while True:
                    gray_frame, faces, pupils, eyes_with_ellipse, is_left_eye_closed, is_right_eye_closed = det.get_features(algorithm_choice)
                    det.full_preview(gray_frame, faces, pupils, eyes_with_ellipse)
                    
                    left_gaze = left_mapper.predict()
                    right_gaze = right_mapper.predict()
                    point = renderer.save_gaze_points(left_gaze, right_gaze)
                    
                    if point is not None:
                        gaze_point_list.append(point)                
                    
                    renderer.render(left_gaze=left_gaze if left_gaze != (None, None) else None,
                                    right_gaze=right_gaze if right_gaze != (None, None) else None,
                                    is_left_eye_closed=is_left_eye_closed,
                                    is_right_eye_closed=is_right_eye_closed)
                                                            
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
            case "3":
                print("\nWybrano: TRYB PODGLĄDU (bez kalibracji i gaze mappingu)\n")
                time.sleep(1)

                while True:
                    gray_frame, faces, pupils, eyes_with_ellipse, is_left_eye_closed, is_right_eye_closed = det.get_features(algorithm_choice)
                    det.full_preview(gray_frame, faces, pupils, eyes_with_ellipse)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            case "4":
                print("\nWybrano: TEST SEKWENCYJNY (6 punktów + mrugnięcie)\n")
                time.sleep(1)
            
                left_mapper = GazeMapper("left")
                left_mapper.load_calibration("calibration_data_left.npz")

                right_mapper = GazeMapper("right")
                right_mapper.load_calibration("calibration_data_right.npz")

                renderer = GazeRenderer()                

                print("Instrukcja:")
                print("- Na ekranie pojawi się 6 zielonych kół")
                print("- Skieruj wzrok na koło")
                print("- ZATWIERDŹ punkt MRUGNIĘCIEM (jedno oko zamknięte)")
                print("- Test zakończy się automatycznie po 6 punktach\n")
                time.sleep(3)
                results = None
                
                while results is None:
                    gray_frame, faces, pupils, eyes_with_ellipse, is_left_eye_closed, is_right_eye_closed = det.get_features(algorithm_choice)

                    det.full_preview(gray_frame, faces, pupils, eyes_with_ellipse)

                    left_gaze = left_mapper.predict()
                    right_gaze = right_mapper.predict()

                    renderer.render(
                        left_gaze=left_gaze if left_gaze != (None, None) else None,
                        right_gaze=right_gaze if right_gaze != (None, None) else None,
                        is_left_eye_closed=is_left_eye_closed,
                        is_right_eye_closed=is_right_eye_closed)

                    results = renderer.simulate_sequence(is_left_eye_closed=is_left_eye_closed, is_right_eye_closed=is_right_eye_closed)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Przerwano przez użytkownika")
                        break

                renderer.pygame_quit()
                cv2.destroyAllWindows()
 
                if results is not None:
                    print("\n=== KONIEC TESTU ===")
                    for i, t in enumerate(results, start=1):
                        print(f"Czas dla koła {i}: {t:.3f} s")
                    renderer.plot_sequence_results(results, filename="result_chart.png")
                    
            case "5":
                print("\nWybrano: TEST PRECYZJI\n")
                time.sleep(1)

                left_mapper = GazeMapper("left")
                left_mapper.load_calibration("calibration_data_left.npz")

                right_mapper = GazeMapper("right")
                right_mapper.load_calibration("calibration_data_right.npz")

                renderer = GazeRenderer()

                gaze_point_list = []

                while True:
                    gray_frame, faces, pupils, eyes_with_ellipse, is_left_eye_closed, is_right_eye_closed = (
                        det.get_features(algorithm_choice)
                    )


                    left_gaze = left_mapper.predict()
                    right_gaze = right_mapper.predict()

        
                    point = renderer.save_gaze_points(left_gaze, right_gaze)
                    if point is not None:
                        gaze_point_list.append(point)

                    renderer.render(
                        left_gaze=left_gaze if left_gaze != (None, None) else None,
                        right_gaze=right_gaze if right_gaze != (None, None) else None,
                        is_left_eye_closed=is_left_eye_closed,
                        is_right_eye_closed=is_right_eye_closed,
                    )

        
                    gaze_available = (
                        left_gaze != (None, None)
                        or right_gaze != (None, None)
                    )

                    result = renderer.precision_test(gaze_available)

                    if result is not None:
                        duration = renderer.precision_duration

                        fs = len(gaze_point_list) / duration if duration > 0 else 0

                        print("Test zakończony.")
                        print("Punkt referencyjny:", result)
                        print("Liczba próbek:", len(gaze_point_list))
                        print(f"Częstotliwość próbkowania: {fs:.2f} Hz")

                        break

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                
                renderer.plot_precision_results(
                    gaze_points=gaze_point_list,
                    target_point=result,
                    sampling_frequency=fs,
                    filename="precision_result.png"
                    )

                renderer.save_precision_results(
                    gaze_points=gaze_point_list,
                    target_point=result,
                    sampling_frequency=fs,
                    filename="precision_result.txt"
                    )
                
                renderer.pygame_quit()
            
            case _:            
                print("Nieznana opcja")
                time.sleep(1)        
        
    except KeyboardInterrupt:
        print("\nPrzerwano przez użytkownika")
    except Exception as e:
        print(f"Błąd: {e}")
    finally:
        det.stop()
