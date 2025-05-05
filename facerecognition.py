import cv2
import numpy as np
import face_recognition
import os
import pickle
from datetime import datetime
from collections import deque

class FaceRecognitionSystem:

    def __init__(self):
        
        self.faces_dir = "known_faces"
        os.makedirs(self.faces_dir, exist_ok=True)
        
        # face encodings
        self.encodings_file = "face_encodings.pkl"
        
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces()
        
        # Inicializace kamery
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Cannot open camera")
            exit()
            
        #Nacteni kaskady pro detekci4 obliceje
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # sledovni tvari mezi snimky(redukce blikání identifikace)
        self.face_trackers = {}  # Slovník pro sledování tváří
        self.next_face_id = 0
        self.max_frames_to_track = 30  # redukce snimku pro plynulost
        
        # training
        self.training_mode = False
        self.captured_faces = []
        self.current_name = ""
        self.min_training_faces = 5 #pocet snimku tvare
        
        # parametry pro rozpoznavani
        self.recognition_threshold = 0.6  #nizsi prah = prisnejsi porovnani
        self.recognition_frequency = 5  
        self.frame_count = 0
        
        print("Face Recognition System Initialized")
        print("Press 's' to start saving a new person's face")
        print("Press 'q' to quit")
    
    def load_known_faces(self):
        """
        Načte dříve uložená kódování tváří ze souboru.
        Tato metoda je volána při inicializaci systému.
        """
        if os.path.exists(self.encodings_file):
            try:
                with open(self.encodings_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data.get('encodings', [])
                    self.known_face_names = data.get('names', [])
                print(f"Loaded {len(self.known_face_names)} known faces")
            except Exception as e:
                print(f"Error loading face encodings: {e}")
        else:
            print("No saved faces found. Start with training mode.")
    
    def save_known_faces(self):
        """
        Uloží kódování tváří do souboru pro pozdější použití.
        Volá se po přidání nové osoby do databáze.
        """
        data = {
            'encodings': self.known_face_encodings,
            'names': self.known_face_names
        }
        with open(self.encodings_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved {len(self.known_face_names)} face encodings")
    
    def enter_training_mode(self):
        """
        Aktivuje trénovací režim pro zachycení nové tváře.
        Zobrazí GUI pro zadání jména osoby a připravuje systém pro zachycení obrazů.
        """
        self.training_mode = True
        self.captured_faces = []
        # zjiskani jmena prez gui
        name = self.get_name_input()
        if not name:  
            self.training_mode = False
            return
        self.current_name = name
        print(f"Training mode activated for {self.current_name}")
        print(f"Capturing {self.min_training_faces} face images. Please move your face slightly between captures.")
    
    def get_name_input(self):
        """
        Zobrazí jednoduché GUI okno pro zadání jména osoby.
        Vrací zadané jméno nebo prázdný řetězec při zrušení.
        """
        # vytvoreni prazdneho obrazu pro vstupni okno 
        input_img = np.zeros((200, 400, 3), np.uint8)
        cv2.putText(input_img, "Enter person's name", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(input_img, "Type name and press Enter", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(input_img, "Press Esc to cancel", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        cv2.imshow('Name Input', input_img)
        
        #inicializace jmena usera
        name = ""
        
        # smycka na ziskani jmena 
        while True:
            key = cv2.waitKey(0) & 0xFF
            
            #enter pro potvrzeni
            if key == 13:  #  Enter
                if name:  #nenull jmeno
                    break
            #escape pro zruseni
            elif key == 27:  # Esc
                name = ""
                break
            #smazani znaku backspace
            elif key == 8:  # Backspace
                name = name[:-1]
            # Běžný vstup znaků
            elif 32 <= key <= 126:  # znaky ASCII
                name += chr(key)
            
            # aktualizace zobrazeni s aktualnim jmenem
            input_img_copy = input_img.copy()
            cv2.putText(input_img_copy, name, (10, 150), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow('Name Input', input_img_copy)
        
        cv2.destroyWindow('Name Input')
        return name
    
    def capture_training_face(self, frame, face_location):
        """
        Zachytí obraz tváře pro trénink.
        Uloží výřez s tváří do adresáře a přidá do seznamu zachycených tváří.
        
        Args:
            frame: Aktuální snímek z kamery
            face_location: Pozice tváře (top, right, bottom, left)
        """
        top, right, bottom, left = face_location
        face_image = frame[top:bottom, left:right]
        
        # ulozeni tvare do adresare
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join(self.faces_dir, f"{self.current_name}_{timestamp}.jpg")
        cv2.imwrite(filename, face_image)
        
        self.captured_faces.append(face_image)
        print(f"Captured face {len(self.captured_faces)}/{self.min_training_faces}")
        
        # kontrola jestli je dostatek zachycenych tvari
        if len(self.captured_faces) >= self.min_training_faces:
            self.process_training_faces()
    
    def process_training_faces(self):
        """
        Zpracuje zachycené tváře a přidá je do databáze známých tváří.
        Vypočítá průměrné kódování tváře pro lepší rozpoznávání.
        """
        print("Processing captured faces...")
        
        # prevod na kodovani tvari
        new_encodings = []
        
        for face_image in self.captured_faces:
            # prevod z BGR na RGB
            rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # Nalezení kódování tváře (může být prázdné, pokud tvář není zřetelná)
            encodings = face_recognition.face_encodings(rgb_face)
            if encodings:
                new_encodings.append(encodings[0])
        
        if new_encodings:
            # prumerne kodovani pro lepsi rozpoznavani
            avg_encoding = np.mean(new_encodings, axis=0)
            
            # Přidání do známých tváří
            self.known_face_encodings.append(avg_encoding)
            self.known_face_names.append(self.current_name)
            
            # Uložení do souboru
            self.save_known_faces()
            
            print(f"Successfully added {self.current_name} to known faces")
        else:
            print("Could not find clear faces in the captured images")
        
        self.training_mode = False
    
    def track_faces(self, face_locations, face_names=None):
        """
        Sleduje tváře mezi snímky pro omezení blikání identifikace.
        Přiřazuje každé tváři ID a sleduje její pohyb mezi snímky.
        
        Args:
            face_locations: Seznam pozic tváří
            face_names: Seznam jmen odpovídajících tvářím (volitelné)
            
        Returns:
            Slovník ID aktuálně sledovaných tváří
        """
        current_face_ids = {}
        
        # Přiřazení ID k aktuálně detekovaným tvářím
        for i, (top, right, bottom, left) in enumerate(face_locations):
            face_center = ((left + right) // 2, (top + bottom) // 2)
            
            # Kontrola, zda je tato tvář blízko existující sledované tváře
            matched_id = None
            for face_id, data in self.face_trackers.items():
                last_center = data['center']
                # Výpočet vzdálenosti mezi středy
                distance = np.sqrt((face_center[0] - last_center[0])**2 + 
                                  (face_center[1] - last_center[1])**2)
                
                # Pokud je dostatečně blízko, pravděpodobně jde o stejnou tvář
                if distance < 50:  
                    matched_id = face_id
                    break
            
            # Pokud není nalezena shoda, přiřadit nové ID
            if matched_id is None:
                matched_id = self.next_face_id
                self.next_face_id += 1
                self.face_trackers[matched_id] = {
                    'center': face_center,
                    'name_history': deque(maxlen=10),
                    'frames_tracked': 0,
                    'bbox': (top, right, bottom, left)
                }
            
            # Aktualizace stávajícího sledovače
            else:
                self.face_trackers[matched_id]['center'] = face_center
                self.face_trackers[matched_id]['frames_tracked'] += 1
                self.face_trackers[matched_id]['bbox'] = (top, right, bottom, left)
            
            # Pokud máme jména pro tento snímek, aktualizovat historii jmen
            if face_names and i < len(face_names):
                self.face_trackers[matched_id]['name_history'].append(face_names[i])
            
            current_face_ids[matched_id] = True
        
        # Odstranění sledovačů pro tváře, které již nejsou viditelné
        face_ids_to_delete = []
        for face_id in self.face_trackers:
            if face_id not in current_face_ids:
                self.face_trackers[face_id]['frames_tracked'] -= 1
                # Odstranění tváří, které nebyly viděny delší dobu
                if self.face_trackers[face_id]['frames_tracked'] <= 0:
                    face_ids_to_delete.append(face_id)
        
        for face_id in face_ids_to_delete:
            del self.face_trackers[face_id]
        
        return current_face_ids
    
    def get_stable_name(self, face_id):
        """
        Získá nejčastější jméno z historie pro omezení blikání.
        Používá "hlasování" z posledních několika snímků.
        
        Args:
            face_id: ID sledované tváře
            
        Returns:
            Nejstabilnější jméno nebo "Unknown"
        """
        if face_id not in self.face_trackers:
            return "Unknown"
        
        name_history = self.face_trackers[face_id]['name_history']
        if not name_history:
            return "Unknown"
        
        # Počítání výskytů každého jména
        name_counts = {}
        for name in name_history:
            if name in name_counts:
                name_counts[name] += 1
            else:
                name_counts[name] = 1
        
        # Nalezení nejčastějšího jména
        most_common_name = max(name_counts, key=name_counts.get)
        
        # Použití pouze pokud se vyskytuje ve více než 30 % snímků
        if name_counts[most_common_name] / len(name_history) > 0.3:
            return most_common_name
        
        return "Unknown"
    
    def run(self):
        """
        Hlavní smyčka pro rozpoznávání tváří.
        Zpracovává snímky z kamery, detekuje tváře a zobrazuje výsledky.
        """
        while True:
            # Načtení snímku z kamery
            ret, frame = self.cap.read()
            if not ret:
                print("Cannot read frame")
                break
            
            # Vytvoření kopie snímku pro zobrazení
            display_frame = frame.copy()
            
            # Zvýšení počítadla snímků
            self.frame_count += 1
            
            # Změna velikosti snímku pro rychlejší rozpoznávání tváří
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Vždy detekovat pozice tváří (je to rychlejší než rozpoznávání)
            face_locations = face_recognition.face_locations(rgb_small_frame)
            
            # Změna měřítka pozic tváří zpět na původní velikost
            scaled_face_locations = []
            for top, right, bottom, left in face_locations:
                scaled_face_locations.append((
                    top * 4, right * 4, bottom * 4, left * 4
                ))
            
            # Zpracování rozpoznávání méně často pro zlepšení výkonu
            do_recognition = (self.frame_count % self.recognition_frequency == 0)
            
            # Zpracování rozpoznávání tváří na vybraných snímcích a když není v trénovacím režimu
            if do_recognition and not self.training_mode and self.known_face_encodings:
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                
                face_names = []
                for face_encoding in face_encodings:
                    # Porovnání se známými tvářemi
                    name = "Unknown"
                    
                    # Pokusit se najít shodu jen pokud máme známé tváře
                    if self.known_face_encodings:
                        # Výpočet vzdálenosti ke všem známým tvářím
                        face_distances = face_recognition.face_distance(
                            self.known_face_encodings, face_encoding
                        )
                        
                        # Nalezení nejlepší shody (nejmenší vzdálenost)
                        best_match_index = np.argmin(face_distances)
                        
                        # Považujeme za shodu pouze pokud je vzdálenost pod prahem
                        if face_distances[best_match_index] < self.recognition_threshold:
                            name = self.known_face_names[best_match_index]
                    
                    face_names.append(name)
                
                # Sledování tváří a aktualizace historie jmen
                current_ids = self.track_faces(scaled_face_locations, face_names)
            else:
                # Pouze sledování pozic tváří bez rozpoznávání
                current_ids = self.track_faces(scaled_face_locations)
            
            # Pokud je v trénovacím režimu a je detekována tvář
            if self.training_mode and scaled_face_locations and len(scaled_face_locations) == 1:
                # Pro trénink zachytit pouze jednu tvář najednou
                top, right, bottom, left = scaled_face_locations[0]
                
                # Vykreslení obdélníku s jinou barvou pro trénink
                cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 165, 255), 2)
                
                # Zobrazení zprávy o tréninku
                cv2.putText(display_frame, f"Capturing {len(self.captured_faces)+1}/{self.min_training_faces}", 
                          (left, top - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 165, 255), 1)
                
                # Přidat tvář pouze při stisku mezerníku
                if cv2.waitKey(1) & 0xFF == 32:  # Mezerník
                    self.capture_training_face(frame, (top, right, bottom, left))
            
            # Zobrazení všech sledovaných tváří s jejich stabilními jmény
            elif not self.training_mode:
                for face_id in current_ids:
                    top, right, bottom, left = self.face_trackers[face_id]['bbox']
                    
                    # Získání stabilního jména pro tuto tvář
                    name = self.get_stable_name(face_id)
                    
                    # Nastavení barvy podle stavu rozpoznávání
                    color = (0, 255, 0) if name != "Unknown" else (255, 0, 0)
                    
                    # Vykreslení obdélníku kolem tváře
                    cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
                    
                    # Vykreslení popisku se jménem
                    cv2.rectangle(display_frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                    cv2.putText(display_frame, name, (left + 6, bottom - 6), 
                               cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
            
            # Zobrazení stavu na displeji
            status_text = "Press 's' to save a new face | Press 'q' to quit" if not self.training_mode else \
                         f"Training mode: {self.current_name} - Press SPACE to capture face ({len(self.captured_faces)}/{self.min_training_faces})"
            cv2.putText(display_frame, status_text, (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            
            # Zobrazení snímku
            cv2.imshow('Face Recognition', display_frame)
            
            # Kontrola stisknutých kláves
            key = cv2.waitKey(1) & 0xFF
            
            # Stisknutí 'q' pro ukončení
            if key == ord('q'):
                break
            
            # Stisknutí 's' pro spuštění trénovacího režimu
            elif key == ord('s') and not self.training_mode:
                self.enter_training_mode()
        
        # Úklid prostředků
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
     # Vytvoření a spuštění instance systému rozpoznávání tváří
     face_system = FaceRecognitionSystem()
     face_system.run()