import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage as ski

#1.1
print("-----------Aufgabe 1.1-------------")
x = 1
y = 1
z = x + y
print("z =", z, "\n")


#1.2
print("-----------Aufgabe 1.2-------------")
# (In einem Skript werden die Anweisungen einfach nacheinander ausgeführt.)
x = 1
y = 1
z = x + y
print("z =", z, "\n")


#1.3
print("-----------Aufgabe 1.3-------------")
print("Aktuelles Arbeitsverzeichnis:", os.getcwd())

ordnername = "Lektion01" 
zielpfad = os.path.join(os.getcwd(), ordnername)
print("Beispielpfad mit os.path.join():", zielpfad)

# wechseln, falls der Ordner existiert damit das Skript nicht crasht
if os.path.exists(zielpfad):
    os.chdir(zielpfad)
    print("Gewechselt nach:", os.getcwd())
else:
    print("Ordner existiert nicht -> kein os.chdir() durchgeführt.\n")


#1.4
print("-----------Aufgabe 1.4-------------")
fname = "roentgen_thorax.jpg"
I = cv2.imread(fname, cv2.IMREAD_UNCHANGED)

if I is None:
    print(f"Fehler: '{fname}' konnte nicht geladen werden.\n")
else:
    print("I.shape:", I.shape)
    print("I.dtype:", I.dtype, "\n")


#1.5
print("-----------Aufgabe 1.5-------------")
fname = "roentgen_thorax.jpg"
I = cv2.imread(fname, cv2.IMREAD_UNCHANGED)

if I is None:
    print(f"Fehler: '{fname}' konnte nicht geladen werden.\n")
else:
    plt.figure()
    # Röntgenbild ist typischerweise Graustufen:
    if I.ndim == 2:
        plt.imshow(I, cmap="gray")
    else:
        # falls doch Farbe: BGR -> RGB für matplotlib
        I_rgb = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
        plt.imshow(I_rgb)
    plt.axis("off")
    plt.title("roentgen_thorax.jpg")
    plt.show()
    print()


#1.6
print("-----------Aufgabe 1.6-------------")
fname = "roentgen_kiefer.jpg"
I = cv2.imread(fname, cv2.IMREAD_UNCHANGED)

if I is None:
    print(f"Fehler: '{fname}' konnte nicht geladen werden.\n")
else:
    if I.ndim == 3:
        I_rgb = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
        I_gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(I_rgb)
        plt.axis("off")
        plt.title("Farb-Bild")

        plt.subplot(1, 2, 2)
        plt.imshow(I_gray, cmap="gray")
        plt.axis("off")
        plt.title("Grauwertbild")

        plt.tight_layout()
        plt.show()
    else:
        print("Bild ist bereits ein Grauwertbild.\n")


#1.7
print("-----------Aufgabe 1.7-------------")
in_name = "roentgen_kiefer.jpg"
out_name = "roentgen_kiefer_grayscale.jpg"

I = cv2.imread(in_name, cv2.IMREAD_UNCHANGED)
if I is None:
    print(f"Fehler: '{in_name}' konnte nicht geladen werden.\n")
else:
    if I.ndim == 3:
        I_gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    else:
        I_gray = I

    ok = cv2.imwrite(out_name, I_gray)
    if ok:
        print(f"Gespeichert als '{out_name}'\n")
    else:
        print("Fehler beim Speichern.\n")

#1.8
print("-----------Aufgabe 1.8-------------")
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

print("A:\n", A)
print("Datentyp von A:", A.dtype)

A8 = A.astype(np.uint8)
print("\nA8:\n", A8)
print("Datentyp von A8:", A8.dtype, "\n\n")

#1.9
print("-----------Aufgabe 1.9-------------")
E = np.eye(3)
Z = np.zeros((2,4))
O = np.ones((2,3))
R = np.random.rand(2,4)

print("eye:\n" , E, E.shape, E.dtype, "\n")
print("zeros:\n" , Z, Z.shape, Z.dtype, "\n")
print("ones:\n" , O, O.shape, O.dtype, "\n")
print("random:\n" , R, R.shape, R.dtype, "\n")

#1.10
print("-----------Aufgabe 1.10-------------")

A = np.array([[1,2,3], [4,5,6], [7,8,9]])
print(A, "\n")
print(A[1, 2], "\n")
print(A[2, :], "\n")
print(A[:, 0], "\n")
print(A[0:2, 0:2], "\n")
A[0, 0] = 9
print(A, "\n")
A[1:3, 1:3] = 1
print(A, "\n")
A = np.delete(A, 2, axis=1)
print(A, "\n")
A = np.insert(A, 2, [1,2,3], axis=1)
print(A)
           

#1.11
print("-----------Aufgabe 1.11-------------")

A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]], dtype=int)
print(A, "\n")

B = A + 2
print(B, "\n")

B = A * 2
print(B, "\n")

B = A * A
print(B, "\n")

E = np.eye(3, dtype=A.dtype)
B = A*E
print(B, "\n")


#1.12
print("-----------Aufgabe 1.12-------------")
fname = "mri_colored.jpg"
I = cv2.imread(fname)

if I is None:
    print(f"Fehler beim lesen des Bildes '{fname}'")
else:
    if I.ndim == 3:
        print("Farbbild Erkannt -> Konvertiere zu Graubild . . .")
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(I)
        plt.axis('off')
        I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
        plt.subplot(1, 2, 2)
        plt.imshow(I, cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
    else:
        print("Kein Farbbild (vermutlich bereits Grauwertbild).")


    # Zeilen, Spalten, Größe ausgeben
    zeilen, spalten = A.shape[0], A.shape[1]
    print(f"Zeilen: {zeilen}")
    print(f"Spalten: {spalten}")
    print(f"Dimension (A.ndim): {A.ndim}")
    print(f"Form (A.shape): {A.shape}")
    print(f"Größe der Bildmatrix (A.size = Anzahl Elemente): {A.size}")

