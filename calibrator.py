import pygame
import numpy as np
import globals
import time
from constants import (
    const_outlier_removal_threshold,
    const_median_absolute_deviation_coeff
)
import os


class CalibrationCollector:
    def __init__(self):

#         self.rows = 5      
#         self.cols = 5      
#         self.margin = 0.05 
# 
#         xs = np.linspace(self.margin, 1.0 - self.margin, self.cols)
#         ys = np.linspace(self.margin, 1.0 - self.margin, self.rows)
# 
#         normalized_points = [(x, y) for y in ys for x in xs]

        normalized_points = [
            (0.05, 0.05), (0.95, 0.5), (0.5, 0.95), (0.05, 0.5), (0.95, 0.05),
            (0.5, 0.05), (0.95, 0.95), (0.05, 0.95), (0.5, 0.5),
            (0.7, 0.3), (0.3, 0.7), (0.7, 0.7), (0.3, 0.3),
            (0.8, 0.4), (0.6, 0.8), (0.2, 0.6), (0.4, 0.2),
            (0.8, 0.6), (0.4, 0.8), (0.2, 0.4), (0.6, 0.2),
            (0.85, 0.15), (0.15, 0.85), (0.15, 0.15), (0.85, 0.85)
            ]

        self.calibration_points = [(int(x * globals.screen_width), int(y * globals.screen_height)) for x, y in normalized_points]
        self.dot_radius = 16
        self.dot_duration = 3
        self.points = self.calibration_points
       
        self.X_left = []
        self.X_right = []
        self.y_left = []
        self.y_right = []
        
        self._current_point_samples = []
        self.current_point_index = 0
        self.dot_visible = False
        self.dot_position = None
        self.dot_start_time = None

        self.samples = 0
        self.samples_discarded = 0
        self.calibration_finished = False

        pygame.init()
        self.screen = pygame.display.set_mode((globals.screen_width, globals.screen_height))
        pygame.display.set_caption("Calibration")        

    def draw_dot(self, position):
        self.screen.fill((0, 0, 0))
        pygame.draw.circle(self.screen, (255, 0, 0), position, self.dot_radius)
        pygame.display.flip()

        self.dot_visible = True
        self.dot_position = position
        self.dot_start_time = time.time()
        self._current_point_samples = []

    def remove_dot(self):
        self._finalize_current_point_samples()

        self.screen.fill((0, 0, 0))
        pygame.display.flip()

        self.dot_visible = False
        self.dot_position = None
        self.dot_start_time = None

        self.current_point_index += 1
        if self.current_point_index >= len(self.points):
            self.calibration_finished = True


    def _collect_features(self):
        left_eye = globals.last_left_vector
        right_eye = globals.last_right_vector

        if left_eye is None or right_eye is None:
            return None, None

        return list(left_eye), list(right_eye)


    def run_calibration(self):
        """
        Purpose:
            Realizuje główną pętlę logiki procesu kalibracji punktowej. 
            Obsługuje zdarzenia Pygame, wyświetla punkty kalibracyjne, 
            pobiera wektory cech oczu, waliduje próbki oraz akumuluje poprawne dane.

        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.pygame_quit()
                exit()

        if self.calibration_finished:
            return

        current_time = time.time()

        if not self.dot_visible:
            self.draw_dot(self.points[self.current_point_index])

        left_eye, right_eye = self._collect_features()
        self.samples += 1

        if (
            left_eye is None
            or right_eye is None
            or any(f == 0 for f in left_eye)
            or any(f == 0 for f in right_eye)
        ):
            self.samples_discarded += 1
            return

        self._current_point_samples.append((left_eye, right_eye))

        if current_time - self.dot_start_time >= self.dot_duration:
            self.remove_dot()


    def _remove_outliers(self, X):
        """
        Purpose:
            Wykonuje detekcję i eliminację obserwacji odstających w zbiorze wektorów cech
            za pomocą mediany i Median Absolute Deviation (MAD).

        Parameters:
            X (np.ndarray):
                Macierz próbek o wymiarach (N, D), gdzie N to liczba próbek, 
                a D to liczba cech wektora oka.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Przefiltrowana macierz próbek.
                - Maska logiczna, gdzie True oznacza próbkę zaakceptowaną, False – odrzuconą.

        """
        if X.size == 0:
            return X, np.zeros(len(X), dtype=bool)

        median_X = np.median(X, axis=0)
        distances_from_median = np.linalg.norm(X - median_X, axis=1)
        median_absolute_deviation = np.median(distances_from_median)

        if median_absolute_deviation == 0:
            mask = np.ones(len(distances_from_median), dtype=bool)
        else:
            robust_z_scores = const_median_absolute_deviation_coeff * distances_from_median / median_absolute_deviation
            mask = robust_z_scores < const_outlier_removal_threshold

        return X[mask], mask


    def _finalize_current_point_samples(self):
        if not self._current_point_samples:
            print(f"[Punkt {self.current_point_index + 1}] Brak próbek.")
            return

        left_samples = np.array([s[0] for s in self._current_point_samples])
        right_samples = np.array([s[1] for s in self._current_point_samples])

        print("\n")
        print(f"Punkt {self.current_point_index + 1}")
        print(f"Łącznie zebrano: {2*(len(self._current_point_samples))} próbek")

        left_filtered, left_mask = self._remove_outliers(left_samples)
        right_filtered, right_mask = self._remove_outliers(right_samples)

        print(f"Lewe oko:")
        print(f"  przyjęte: {len(left_filtered)} z {len(left_samples)}")
        print(f"  odrzucone (outliery): {np.sum(~left_mask)}")

        print(f"Prawe oko:")
        print(f"  przyjęte: {len(right_filtered)} z {len(right_samples)}")
        print(f"  odrzucone (outliery): {np.sum(~right_mask)}")

        for s in left_filtered:
            self.X_left.append(s.tolist())
            self.y_left.append(list(self.points[self.current_point_index]))

        for s in right_filtered:
            self.X_right.append(s.tolist())
            self.y_right.append(list(self.points[self.current_point_index]))

        print("Dane zapisane.")
        print("\n")

        self._current_point_samples = []


    def is_calibration_ready(self):
        return self.calibration_finished
    
    def save_calibration(self):
 
        left_filename = "calibration_data_left.npz"
        right_filename = "calibration_data_right.npz"

        np.savez(
            left_filename,
            X_left = np.array(self.X_left),
            y = np.array(self.y_left))

        np.savez(
            right_filename,
            X_right = np.array(self.X_right),
            y = np.array(self.y_right))

        print(f"\nDane kalibracyjne zapisane:")
        print(f"  - lewe oko : {left_filename}  ({len(self.X_left)} próbek)")
        print(f"  - prawe oko: {right_filename} ({len(self.X_right)} próbek)\n")

    def does_calibration_file_exist(filename):
        return os.path.isfile(filename)

    def pygame_quit(self):
        pygame.display.quit()
        pygame.quit()
