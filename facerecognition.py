import cv2
#Spustitelne pomoci prikazu ' python3 <misto souboru> '
# Otevre prvni pripojenou kameru, prvni byva na indexu


##Pouzijte pokud chcete vzit video z kamery z Rpi ci jine, popripade nutno zmenit '0' na jiny vstup
#cap = cv2.VideoCapture(0)


##Opencv predem trenovany model na rozpoznani zakladnich tvaru obliceje
#Ve slozce data.haarcascades muzeme najit vice modelu pro vice ucelu, my ale pouzijeme na detekci obliceje
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


#Pouzijte pokud chcete video vzit ze souboru
cap = cv2.VideoCapture("Video/video2.mp4")


age_net = cv2.dnn.readNetFromCaffe("age_deploy.prototxt", "age_net.caffemodel")

AGE_RANGES = ['(0-3)', '(4-6)', '(7-9)', '(10-12)', '(13-15)', '(16-18)', '(19-21)', '(22-24)', '(25-27)', '(28-30)', 
              '(31-33)', '(34-36)', '(37-39)', '(40-42)', '(43-45)', '(46-48)', '(49-51)', '(52-54)', '(55-57)', 
              '(58-60)', '(61-63)', '(64-66)', '(67-69)', '(70-72)', '(73-75)', '(76-78)', '(79-81)', '(82-84)', 
              '(85-87)', '(88-90)', '(91-93)', '(94-96)', '(97-99)', '(100+)']

# Jestli se kamera neotevrela spravne
if not cap.isOpened():
    print("Nelze otevřít kameru")
    exit()

# Cyklus pro cteni a potom zobrazeni snimku
while True:
    
    # Cteni snimku z kamery
    ret, frame = cap.read()
    
    # Pokud neco selze
    if not ret:
        print("Nelze přečíst snímek")
        break
     
    ##Prevedeme obraz z RGB na stupne sedi, Model lepe funguje na rozpoznani obliceje v barvach sedi
    # cv2.imshow('Camera - Face Detection', gray_frame) -> takto model vidi vstupni obraz
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    
    ## metoda '.detectMultiScale(vstupni obraz, meritko, pocet sousednich obdelniku)
    # vraci seznam obdelniku, kazdy obdelnik je jedna detekovana oblast(oblicej)
    # kazdy ten obdelnik ma souradnice (x,y,w,h)
    ## scaleFactor -> Ridi jak moc se velikost obrazu zmensi pri kazdem pruchodu algoritmu
    # kazdy obraz zmensi o 10% (puvodne 110% puvodni velikosti), bude hledat objekty v ruznych velikostech
    # scaleFactor(1.005) Vyssi presnost, pomalejsi vykon, scaleFactor(2) Rychlejsi vykon, nizsi presnost obliceje budou prehlednuty
    ## minNeighbors -> Urcuje kolik musi byt sousednich obdelniku, ktere jsou detekovany jako oblicej
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)


    ## cyklus projde vsechny detekovane obliceje, kazdy obdelnik je popsan jako x,y,w,h
    for (x, y, w, h) in faces:
        # x a y -> Levy horni roh obdelniku
        # w a h -> Sirka a vyska obdelniku
        cv2.rectangle(frame, (x, y), (x + h, y + w), (255, 0, 0), 3)
        
        face_img = frame[y:y+h, x:x+w]
        
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = AGE_RANGES[age_preds[0].argmax()]

        
        label = f"Age: {age}"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            
        
    ##Zobrazeni obrazu v okne
    cv2.imshow('Face Detection', frame)
        
    # Po jedne ms a zmacknuti q se okno zavre
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    
#Ukonceni procesu a zavreni okna
cap.release()
cv2.destroyAllWindows()
