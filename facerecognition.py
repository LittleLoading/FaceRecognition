import cv2
#Spustitelne pomoci prikazu ' python3 <misto souboru> '
# Otevre prvni pripojenou kameru, prvni byva na indexu
cap = cv2.VideoCapture(0)


#Nastaveni obdelniku - zatim provizorne
startpoint = (100,50)
endpoint = (125,80)
color = (255,255,255)
thicnes = 2
    
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
    
    #Nastaveni zobrazeni obdelniku
    frame = cv2.rectangle(frame,startpoint,endpoint, color,thicnes)
    # Zobrazeni obrazu v okne s nazvem Camera Feed
    cv2.imshow('Camera Feed', frame)
    
    # Po jedne ms a zmacknuti q se okno zavre
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    
#Ukonceni procesu a zavreni okna
cap.release()
cv2.destroyAllWindows()
