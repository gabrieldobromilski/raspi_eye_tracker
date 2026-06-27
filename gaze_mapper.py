from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.preprocessing import MinMaxScaler
import globals
import math
from collections import deque
import numpy as np

class GazeMapper:
    def __init__(self, eye: str):
        self.eye = eye
        self.model_choice = self.choose_model()

        self.scaler = MinMaxScaler()
        self.model_x, self.model_y = self._init_models()
        self.is_ready = False
        
        self.feature_buffer_size = 5
        self.feature_buffer = deque(maxlen=self.feature_buffer_size)


    def choose_model(self):
        print("\n=== Wybierz model do mappera ===")
        print("1 - Support Vector Regressor")
        print("2 - Random Forest Regressor")
        print("3 - Extra Trees Regressor")
        print("--------------------------------------")
        while True:
            choice = input("Wybierz [1/2/3]: ").strip()
            if choice in ("1", "2", "3"):
                return choice
            print("Niepoprawny wybór. Spróbuj ponownie.\n")


    def _init_models(self):
        match self.model_choice:
            case "1":
                print("GazeMapper: używany SVR")
                return (
                    SVR(kernel="rbf", C=60.0, epsilon=0.03, gamma="scale"),
                    SVR(kernel="rbf", C=60.0, epsilon=0.03, gamma="scale")
                )
            case "2":
                print("GazeMapper: używany RandomForestRegressor")
                return (
                    RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
                    RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
                )
            case "3":
                print("GazeMapper: używany ExtraTreesRegressor")
                return (
                    ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=-1),
                    ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=-1)
                )


    def load_calibration(self, filename):
        """
        Purpose:
            Ładuje dane kalibracyjne z pliku, normalizuje je względem rozdzielczości ekranu,
            skaluje cechy wejściowe i trenuje modele regresji dla osi X i Y. Ustawia flagę is_ready na True po zakończeniu treningu.

        Parameters:
            filename (str): Ścieżka do pliku .npz zawierającego dane kalibracyjne.
        """
        data = np.load(filename)

        if self.eye == "left":
            X = data["X_left"]
            y = data["y"]
        elif self.eye == "right":
            X = data["X_right"]
            y = data["y"]
        else:
            raise ValueError(f"Nieznane oko: {self.eye}")

        if len(X) != len(y):
            raise ValueError(f"Niespójne dane kalibracyjne ({self.eye}): X={len(X)}, y={len(y)}")

        y_norm = np.empty_like(y, dtype=float)
        y_norm[:, 0] = y[:, 0] / globals.screen_width
        y_norm[:, 1] = y[:, 1] / globals.screen_height

        X_scaled = self.scaler.fit_transform(X)

        self.model_x.fit(X_scaled, y_norm[:, 0])
        self.model_y.fit(X_scaled, y_norm[:, 1])

        self.is_ready = True
        print(f"GazeMapper ({self.eye}): model wytrenowany.")


    def _collect_live_features(self):
        if self.eye == "left":
            vec = globals.last_left_vector
        elif self.eye == "right":
            vec = globals.last_right_vector
        else:
            raise ValueError(f"Nieznane oko: {self.eye}")
        
        if vec is None:
            return None

        return np.array(vec, dtype=float)


    def _get_median_features(self, features):
        """
        Purpose:
            Utrzymuje bufor ostatnich N próbek cech oka i zwraca medianę tych próbek,
            redukując szum w danych.

        Parameters:
            features (np.ndarray): Aktualny wektor cech oka.

        Returns:
            np.ndarray: Mediana cech z bufora lub bieżąca próbka, jeśli bufor nie jest pełny.

        """
        self.feature_buffer.append(features)

        if len(self.feature_buffer) < self.feature_buffer_size:
            return features

        stacked = np.vstack(self.feature_buffer)
        return np.median(stacked, axis=0)


    def predict(self):
        """
        Purpose:
            Przewiduje aktualną pozycję spojrzenia na ekranie w pikselach,
            wykorzystując wytrenowane modele regresyjne oraz medianę ostatnich próbek cech.

        Returns:
            Tuple[float, float]:
                - gx, gy: przewidywane współrzędne spojrzenia w pikselach,
        """
        if not self.is_ready:
            return None, None

        features = self._collect_live_features()
        
        if features is None:
            return None, None

        features = self._get_median_features(features)

        X_scaled = self.scaler.transform(features.reshape(1, -1))
        gx_norm, gy_norm = self.model_x.predict(X_scaled)[0], self.model_y.predict(X_scaled)[0]

        if np.isnan(gx_norm) or np.isnan(gy_norm):
            return None, None

        gx = np.clip(gx_norm * globals.screen_width, 0, globals.screen_width)
        gy = np.clip(gy_norm * globals.screen_height, 0, globals.screen_height)

        return float(gx), float(gy)
    

        