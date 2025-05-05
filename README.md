## FACESENSE

**Systém detekce a rozpoznávání tváří pomocí kamery a Raspberry Pi 4B**

**Autor:** David Houra
**Studijní obor:** C

---

## Anotace

Tato práce popisuje návrh, vývoj a testování systému pro detekci a rozpoznávání tváří v reálném čase pomocí kamery Raspberry Pi Camera V2 připojené k platformě Raspberry Pi 4B. Po úspěšné identifikaci a ověření tváře pomocí knihoven OpenCV a face\_recognition je zobrazena vizuální zpětná vazba (ohraničení tváře, zobrazení jména). Systém podporuje interaktivní trénink nových osob a automatické ukládání jejich kódování.

---

## Úvod

V oblasti bezpečnostních a docházkových systémů narůstá využití automatizované detekce a rozpoznávání osob. Cílem tohoto projektu je vytvořit cenově dostupné a modulární řešení založené na otevřeném hardwaru (Raspberry Pi 4B, Raspberry Pi Camera V2) a softwaru (Python, OpenCV, face\_recognition). Mezi klíčové funkce patří:

* Detekce obličejů v reálném čase pomocí Haar Cascade.
* Výpočet face embeddings a porovnání s databází známých tváří.
* Interaktivní režim tréninku nových osob (minimálně 5 snímků).
* Sledování tváří mezi snímky pro omezení blikání labelů.
* Vizuální označení tváří s jejich stabilním jménem nebo "Unknown".

---

## Ekonomická rozvaha

### Analýza konkurence

* Komerční řešení pro rozpoznávání tváří vyžadují drahý hardware a licencovaný software (desítky tisíc Kč).
* Některé open-source projekty nejsou plně integrované nebo vyžadují složitou konfiguraci.

### Výhody našeho projektu

* Nízká cena: \~3 000 Kč za Raspberry Pi 4B a kameru.
* Jednoduchá instalace a konfigurace.
* Využití otevřených knihoven bez licenčních poplatků.

### Způsob propagace

* GitHub repozitář s dokumentací, kódem a ukázkovými daty.
* Prezentace na školních konferencích a workshopech.
* Článek do odborného časopisu o počítačovém vidění.

### Návratnost investic

* Možnost rozšíření pro docházkové a bezpečnostní aplikace.
* Reuse hardware pro další projekty Raspberry Pi, úspory z rozsahu.

---

## Vývoj

### Použité technologie

* **Raspberry Pi 4B:** zpracování videa a řízení systému.
* **Raspberry Pi Camera V2:** snímání obrazu v HD rozlišení.
* **Python 3:** hlavní jazyk projektu.
* **OpenCV:** detekce obličejů Haar Cascade.
* **face\_recognition (dLib):** výpočet a porovnání face embeddings.
* **pickle:** serializace kódování tváří.
* **VNC & SSH:** vzdálený přístup a správa.
* **Git & GitHub:** verzování a synchronizace kódu.

### Architektura a členění kódu

* **main\_pi.py:** inicializace kamery, čtení snímků, detekce, rozpoznávání a vykreslení GUI.
* **face\_data.py:** načítání a ukládání kódování tváří (`face_encodings.pkl`).
* **training.py:** logika interaktivního tréninku a zpracování snímků tváří.
* **tracking.py:** sledování tváří mezi snímky, správa historie jmen.

### Průběh vývoje

1. Vytvoření vývojového prostředí na Linux notebooku (virtualenv).
2. Implementace detekce obličejů v OpenCV a testování na lokálním videu.
3. Integrace face\_recognition pro generování face embeddings.
4. Vývoj interaktivního tréninku: GUI vstup jména, zachytávání 5 snímků.
5. Implementace sledování tváří přes deque a vypočtení středu obličeje.
6. Nastavení vzdáleného přístupu (VNC, SSH) a Git automatizace.
7. Debugging a optimalizace výkonu (úprava rozlišení, paralelizace).

---

## Testování

### Testovací scénáře

1. **Detekce a rozpoznání známé osoby za ideálních světelných podmínek** – úspěšnost ≥ 95 %.
2. **Detekce v horších světelných podmínkách** – úspěšnost ≥ 70 %.
3. **Rozpoznání neznámé osoby** – označení „Unknown“.
4. **Trénink nového uživatele** – správné uložení a rozpoznání ≥ 90 %.
5. **Sledování více osob** – stabilita při 3–4 osobách.
6. **Zátěžový test (1 h běhu)** – stabilita paměti a CPU.

### Výsledky testů

* Scénář 1: 97 % úspěšnost, odezva 100 ms/snímek.
* Scénář 2: 68 % úspěšnost, doporučeno IR osvětlení.
* Scénář 3: 100 % aliasování na „Unknown“.
* Scénář 4: 92 % po 5 tréninkových snímcích.
* Scénář 5: stabilní až 3 osoby, při 5 tvářích FPS < 5.
* Scénář 6: bez paměťových úniků, CPU zatížení 80 %.

---

## Nasazení a spuštění

1. Naformátujte SD kartu a nainstalujte Raspberry Pi OS (32‑bit) s povoleným SSH.
2. Připojte Raspberry Pi Camera V2 a aktivujte ji přes `raspi-config`.
3. Klonujte repozitář: `git clone https://github.com/LittleLoading/FaceRecognition.git`.
4. Na Pi nainstalujte Python 3, OpenCV, face\_recognition, pyvirtualdisplay (pro headless).
5. Upravte `config.json` (cesta k Haar Cascade, threshold, jméno souboru embeddingů).
6. Vytvořte systémovou službu (systemd) pro automatické spuštění:

   ```
   [Unit]
   Description=FaceSense Service
   After=multi-user.target

   [Service]
   ExecStart=/usr/bin/python3 /home/pi/facesense/main_pi.py
   WorkingDirectory=/home/pi/facesense
   Restart=on-failure

   [Install]
   WantedBy=multi-user.target
   ```
7. Aktivujte a spusťte službu: `sudo systemctl enable facesense.service && sudo systemctl start facesense.service`.

**Požadavky:**

* Raspberry Pi 4B s Raspberry Pi Camera V2
* Python 3 a požadované knihovny
* GitHub přístup pro synchronizaci

---

## Licence

Projekt vychází z licence MIT. Viz soubor `LICENSE`.

---

## Odkaz na GitHub

[https://github.com/LittleLoading/FaceRecognition.git](https://github.com/LittleLoading/FaceRecognition.git)

---

## Závěr

Modulární systém pro detekci a rozpoznávání tváří na Raspberry Pi 4B prokázal vysokou přesnost a stabilitu. Projekt je snadno rozšiřitelný o IR osvětlení, GPU akceleraci či integraci s databázemi pro podnikové aplikace.
