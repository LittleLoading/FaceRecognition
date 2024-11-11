##Tahle trida byla vytvorena mnou za ucelem jednodusi manipulace,
##Stale se prepojovat na Rpi abych tam spustil code je zbytecne slozite,
##Od toho je tu tahle trida aby natocila video ktere se ulozi
## a na nem budu svuj model moct vytvaret odkoliv

import cv2

#Spustitelne pomoci prikazu ' python3 <misto souboru> '
# Otevre prvni pripojenou kameru, prvni byva na indexu
cap = cv2.VideoCapture(0)


#Ziskani zakladni width a height obrazu kamery
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))



#Definujici kodek a vytvoreni objektu na zapis videa
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('Video/videoXY.mp4', fourcc, 20.0, (frame_width, frame_height))


# Cyklus pro cteni, zobrazeni, ulozeni snimku
while True: 
    ret, frame = cap.read()
    
    if not ret:
        print("Nelze precist snimek")
        break
    
    ##Zobrazeni snimku v okne
    cv2.imshow('Camera', frame)
    
    #Ulozeni snimku do videa filu
    out.write(frame)
    
    
    #Ukonceni po znmacknuti 'q'
    if cv2.waitKey(1) == ord('q'):
        break

#Ukonceni procesu a zavreni okna
cap.release()
out.release()
cv2.destroyAllWindows()

    