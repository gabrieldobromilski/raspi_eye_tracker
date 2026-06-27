import pygame
import globals
import math
import random
import time
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import numpy as np

class GazeRenderer:
    def __init__(self):
        self.display_initialized = False
        self.screen = None

        self.last_valid_fused_gaze = None
        self.gaze_radius = 10
        
        self.test_1 = [100, 100, 100, 100, 70, 90]
        self.test_2 = [80, 80, 80, 80, 60, 70]
        self.test_3 = [60, 60, 60 ,60, 40, 50]
        
        self.sequence_initialized = False
        self.sequence_index = 0
        self.sequence_positions = []
        self.current_sequence_radius = []
        self.sequence_start_time = None
        self.sequence_times = [None] * 6
        self.sequence_finished = False
        self.button_radius = 100
        self.button_active = False
        self.button_pos = None
        
        self.precision_started = False
        self.precision_initialized = False
        self.precision_finished = False

        self.precision_target = None
        self.precision_radius = 50

        self.precision_duration = globals.test_time
        self.precision_start_time = None

    def _init_display(self):
        pygame.init()
        self.screen = pygame.display.set_mode(
            (globals.screen_width, globals.screen_height)
        )
        pygame.display.set_caption("Gaze Renderer")
        self.display_initialized = True


    def render(self, left_gaze = None, right_gaze = None, is_left_eye_closed = False, is_right_eye_closed = False):
        """
        Purpose:
            Renderuje aktualną scenę spojrzenia na ekranie Pygame. 
            Wyświetla ostatnie poprawne pozycje spojrzenia dla lewego i prawego oka,
            łączy je w przypadku obu widocznych oczu, oraz rysuje aktywny przycisk, jeśli jest włączony.

        Parameters:
            left_gaze (tuple[int, int]): Współrzędne spojrzenia lewego oka w pikselach.
            right_gaze (tuple[int, int]): Współrzędne spojrzenia prawego oka w pikselach.
            is_left_eye_closed (bool): Flaga informująca, czy lewe oko jest zamknięte.
            is_right_eye_closed (bool): Flaga informująca, czy prawe oko jest zamknięte.

        Returns:
            bool: True jeśli renderowanie przebiegło poprawnie, False jeśli Pygame zostało zamknięte.
        """
        if not self.display_initialized:
            self._init_display()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit()
                return False

        self.screen.fill((0, 0, 0))

  
        if self.button_active and self.button_pos:
            pygame.draw.circle(
                self.screen, (0, 255, 0),
                self.button_pos, self.button_radius
            )

        left_gaze_pos = left_gaze if (left_gaze and not is_left_eye_closed) else None
        right_gaze_pos = right_gaze if (right_gaze and not is_right_eye_closed) else None


        left_visible = left_gaze is not None and not is_left_eye_closed
        right_visible = right_gaze is not None and not is_right_eye_closed

        if left_visible and right_visible:
            fused = self.gaze_fusion(left_gaze_pos, right_gaze_pos)
            pygame.draw.circle(
                self.screen, (255, 0, 255),
                (int(fused[0]), int(fused[1])),
                self.gaze_radius
            )
        elif left_visible:
            self.last_valid_fused_gaze = left_gaze_pos
            pygame.draw.circle(
                self.screen, (255, 0, 0),
                (int(left_gaze_pos[0]), int(left_gaze_pos[1])),
                self.gaze_radius
            )
        elif right_visible:
            self.last_valid_fused_gaze = right_gaze_pos
            pygame.draw.circle(
                self.screen, (0, 0, 255),
                (int(right_gaze_pos[0]), int(right_gaze_pos[1])),
                self.gaze_radius
            )

        pygame.display.flip()
        return True


    def gaze_fusion(self, left_gaze=None, right_gaze=None):
        if left_gaze is not None and right_gaze is not None:
            fused_x = (left_gaze[0] + right_gaze[0]) / 2
            fused_y = (left_gaze[1] + right_gaze[1]) / 2
            return fused_x, fused_y
        elif left_gaze is not None:
            return left_gaze
        elif right_gaze is not None:
            return right_gaze
        else:
            return None

    def pygame_quit(self):
        pygame.display.quit()
        pygame.quit()


    def simulate_sequence(self, is_left_eye_closed, is_right_eye_closed):
        """
        Purpose:
            Symuluje gaszenie przycisków na ekranie, rejestrując czas potrzebny na zgaszenie.
            Przycisk jest gaszony jeżeli pozycja spojrzenia pokrywa się z kołem a użytkownik mrugnie.
            Obsługuje inicjalizację sekwencji, aktywację przycisków i przejście do kolejnych etapów.

        Parameters:
            is_left_eye_closed (bool): Flaga informująca, czy lewe oko jest zamknięte.
            is_right_eye_closed (bool): Flaga informująca, czy prawe oko jest zamknięte.

        Returns:
            tuple[float, ...]:
                - Krotka czasów reakcji dla wszystkich przycisków, jeśli sekwencja została zakończona,
        """
        if self.sequence_finished:
            return tuple(self.sequence_times)

     
        if not self.sequence_initialized:

            margin = self.button_radius + 10
            positions = [
                (globals.screen_width - margin, margin),     
                (margin, margin),          
                (margin, globals.screen_height - margin),     
                (globals.screen_width - margin, globals.screen_height - margin),
                (globals.screen_width // 2, globals.screen_height // 2),        
                (random.randint(margin, globals.screen_width - margin), random.randint(margin, globals.screen_height - margin))]
            
            self.sequence_positions = positions
            
            test_button_radiuses = self.test_1
            
            if test_button_radiuses is None:
                self.current_sequence_radius = [self.button_radius] * 6
            else:
                
                radius_list = list(test_button_radiuses)
                if len(radius_list) < 6:
                    radius_list += [self.button_radius] * (6 - len(radius_list))
                self.current_sequence_radius = radius_list[:6]

            self.sequence_index = 0
            self.sequence_start_time = time.time()
            self.sequence_times = [None] * 6
            self.sequence_initialized = True
            self.sequence_finished = False

        idx = self.sequence_index
        pos = self.sequence_positions[idx]
        radius = self.current_sequence_radius[idx]
        self.button_active = True
        self.button_pos = pos
        self.button_radius = radius

        if self.last_valid_fused_gaze is not None:
            last_gaze_x, last_gaze_y = self.last_valid_fused_gaze
        else:
            last_gaze_x, last_gaze_y = None, None
     
        is_blink = bool(is_left_eye_closed or is_right_eye_closed)
        print(f"Blinked: {is_blink}")
       
        if is_blink:
            hit = False
            if last_gaze_x is not None and last_gaze_y is not None:
                if math.hypot(last_gaze_x - pos[0], last_gaze_y - pos[1]) <= radius:
                    hit = True

            if hit:
                elapsed = time.time() - self.sequence_start_time
                self.sequence_times[idx] = elapsed

                self.sequence_index += 1
                if self.sequence_index >= 6:
                    self.sequence_finished = True
                    self.button_active = False
                    return tuple(self.sequence_times)
                else:
                    self.sequence_start_time = time.time()
                    next_pos = self.sequence_positions[self.sequence_index]
                    next_radius = self.current_sequence_radius[self.sequence_index]
                    self.button_pos = next_pos
                    self.button_radius = next_radius
                    return None
            else:
                return None

        return None
    
    def precision_test(self, gaze_available):
        """
        Test precyzji uruchamia się dopiero gdy system wykryje gaze point.
        """
    
        if not gaze_available:
            return None

        if not self.precision_initialized:
            margin = self.precision_radius + 10

            self.precision_target = (
                random.randint(margin, globals.screen_width - margin),
                random.randint(margin, globals.screen_height - margin)
            )

            self.button_active = True
            self.button_pos = self.precision_target
            self.button_radius = self.precision_radius

            self.precision_start_time = time.perf_counter()
            self.precision_initialized = True
            self.precision_started = True

        elapsed = time.perf_counter() - self.precision_start_time

        if elapsed >= self.precision_duration:
            self.button_active = False
            self.precision_finished = True
            return self.precision_target

        return None
    
    def save_gaze_points(self, left_gaze, right_gaze):
        left_valid = (
            left_gaze is not None
            and left_gaze[0] is not None
            and left_gaze[1] is not None
        )

        right_valid = (
            right_gaze is not None
            and right_gaze[0] is not None
            and right_gaze[1] is not None
        )

        if left_valid and right_valid:
            return (
                (left_gaze[0] + right_gaze[0]) / 2,
                (left_gaze[1] + right_gaze[1]) / 2,
            )

        if left_valid:
            return left_gaze

        if right_valid:
            return right_gaze

        return None
    
    def plot_precision_results(self, gaze_points, target_point, sampling_frequency, filename=None):

        if len(gaze_points) == 0:
            print("Brak danych do wizualizacji.")
            return

        
        data = np.array(gaze_points)

        x = data[:, 0]
        y = data[:, 1]

        plt.figure(figsize=(8, 8))

        plt.scatter(x, y, c='red', s=10, label="Gaze points")

        circle = plt.Circle(
            target_point,
            self.precision_radius,
            fill=False,
            color='blue',
            linewidth=2,
            label="Target area"
        )

        plt.gca().add_patch(circle)

        plt.scatter(
            target_point[0],
            target_point[1],
            c='blue',
            s=50,
            label="Target center"
        )

        plt.gca().set_aspect('equal', adjustable='box')

        plt.xlim(0, globals.screen_width)
        plt.ylim(0, globals.screen_height)
        plt.gca().invert_yaxis()  

        plt.title("Precision Test - Gaze Distribution")
        plt.xlabel("X [px]")
        plt.ylabel("Y [px]")
        plt.legend()
        plt.grid(True)
        
        plt.figtext(0.5, 0.01, f"Częstotliwość próbkowania: {sampling_frequency:.2f} Hz", ha="center", fontsize=10)

        plt.tight_layout(rect=[0, 0.04, 1, 1])

        if filename:
            plt.savefig(filename)
            print(f"Wykres zapisany: {filename}")
        else:
            plt.show()

        plt.close()
    
    def plot_sequence_results(self, results, filename=None):
        times = np.array(results)
        indices = np.arange(1, len(times) + 1)
        mean_time = np.mean(times)

        plt.figure(figsize=(10, 5))
        plt.bar(indices, times, color='skyblue')
        plt.axhline(mean_time, linestyle='-', color='red', linewidth=2, label=f"Średnia: {mean_time:.3f} s")

        plt.xticks(indices)
        plt.xlabel("Numer przycisku w sekwencji")
        plt.ylabel("Czas reakcji [s]")
        plt.title("Czasy reakcji na przyciski")
        plt.legend()
        plt.grid(axis='y', linestyle=':')

        plt.tight_layout()

        if filename:
            plt.savefig(filename)
            print(f"Wykres zapisany do: {filename}")
            plt.close()
        else:
            plt.show()
            
    def save_precision_results(self,gaze_points, target_point, sampling_frequency, filename="precision_results.txt"):

        with open(filename, "w") as file:

            file.write("===== TEST PRECYZJI =====\n\n")

            file.write(
                f"Punkt wzorcowy: "
                f"({target_point[0]:.2f}, {target_point[1]:.2f})\n"
            )

            file.write(
                f"Promień okręgu: {self.precision_radius} px\n"
            )

            file.write(
                f"Częstotliwość próbkowania: "
                f"{sampling_frequency:.2f} Hz\n"
            )

            file.write(
                f"Liczba próbek: {len(gaze_points)}\n\n"
            )

            file.write("--------------------------------------------\n")
            file.write("Nr\tX\tY\n")
            file.write("--------------------------------------------\n")

            for i, point in enumerate(gaze_points, start=1):
                file.write(
                    f"{i}\t"
                    f"{point[0]:.2f}\t"
                    f"{point[1]:.2f}\n"
                )

        print(f"Wyniki zapisano do: {filename}") 
