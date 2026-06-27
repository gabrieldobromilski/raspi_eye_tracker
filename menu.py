class Menu():
    def __init__(self):
        self.choice = None
        
    def mode_choice(self):
        print("\n=== Wybierz tryb pracy ===")
        print("1. Wykonaj nową kalibrację")
        print("2. Eye Tracking przy użyciu obecnego calibration_data.npz")
        print("3. Podgląd wykrywania cech")
        print("4. Symulacja przycisku")
        print("5. Test dokładności")
        print("--------------------------------------")
        while True:
            self.choice = input("Wybierz [1/2/3/4/5]: ").strip()

            if self.choice in ("1", "2", "3", "4", "5"):
                return self.choice
            else:
                print("Niepoprawny wybór. Spróbuj ponownie.\n")
        