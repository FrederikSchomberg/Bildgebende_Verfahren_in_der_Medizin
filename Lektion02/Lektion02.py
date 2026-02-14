import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage as ski
import pydicom
    
#2.1
def a1():
    print("-----------Aufgabe 2.1-----------")

    I = cv2.imread("roentgen_kiefer_grayscale.jpg")
    
    plt.figure(figsize=(10, 5), facecolor="black")
    
    plt.imshow(I, cmap="gray")
    plt.axis('off')
    

    min = np.min(I)
    max = np.max(I)
    mean = np.mean(I)
    std = np.std(I)
    sum = np.sum(I)

    print(f"min: {min:.2f}\nmax: {max:.2f}\nmean: {mean:.2f}\nstd: {std:.2f}\nsum: {sum:.2f}\nmeanSum: ", np.mean(sum))
    plt.show()

#2.2
def a2():
    print("-----------Aufgabe 2.2-----------")
    
    I = cv2.imread("roentgen_kiefer_grayscale.jpg")

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(I, cmap="gray")
    plt.subplot(1, 2, 2)
    plt.hist(I.ravel(), bins="auto")
    plt.tight_layout()
    plt.show()

#2.3
def a3():
    print("-----------Aufgabe 2.3-----------")
    
    I = cv2.imread("roentgen_hand.jpg", cv2.IMREAD_GRAYSCALE)
    I2 = cv2.equalizeHist(I)
    
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(I, cmap="gray")
    plt.subplot(2, 2, 2)
    plt.hist(I.ravel(), bins="auto")
    plt.subplot(2, 2, 3)
    plt.imshow(I2, cmap="gray")
    plt.subplot(2, 2, 4)
    plt.hist(I2.ravel(), bins="auto")
    plt.tight_layout()
    plt.show()
    
#2.4
def a4():
    print("-----------Aufgabe 2.4-----------")

    I_u8  = cv2.imread("roentgen_huefte.jpg", cv2.IMREAD_GRAYSCALE)
    BG_u8 = cv2.imread("roentgen_huefte_bg.jpg", cv2.IMREAD_GRAYSCALE)

    if I_u8 is None or BG_u8 is None:
        print("Fehler: Bilddateien konnten nicht geladen werden (Pfad/Name prüfen).")
        return

    if I_u8.shape != BG_u8.shape:
        print("Fehler: Original und Hintergrund haben nicht die gleiche Größe!")
        print("I :", I_u8.shape)
        print("BG:", BG_u8.shape)
        return

    # in float32 rechnen
    I  = I_u8.astype(np.float32)
    BG = BG_u8.astype(np.float32)

    # BG wird auf Mittelwert 1 normiert, damit die Helligkeit ähnlich bleibt
    BG_norm = BG / np.mean(BG)
    I_corr = I / BG_norm

    # Wertebereich zurück auf [0,255] strecken
    gmin = np.min(I_corr)
    gmax = np.max(I_corr)
    I_stretch = (I_corr - gmin) * (255.0 / (gmax - gmin))

    # clip + zurück zu uint8
    I_out = np.clip(I_stretch, 0, 255).astype(np.uint8)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(I_u8, cmap="gray")
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(BG_u8, cmap="gray")
    plt.title("Hintergrund (BG)")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(I_out, cmap="gray")
    plt.title("Korrigiert")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
    
#2.5
def a5():
    print("-----------Aufgabe 2.5-----------")

    # 1) Phantom laden
    I_u8 = cv2.imread("phantom.jpg", cv2.IMREAD_GRAYSCALE)
    if I_u8 is None:
        print("Fehler: phantom.jpg konnte nicht geladen werden.")
        return

    # 2) in float [0,1] umwandeln (für random noise)
    I = I_u8.astype(np.float32) / 255.0

    # 3) weißes rauschen (gaussian) 
    I_g1 = ski.util.random_noise(I, mode="gaussian", mean=0.0, var=0.001)
    I_g2 = ski.util.random_noise(I, mode="gaussian", mean=0.0, var=0.01)

    # 4) aalt & pepper mit verschiedenen werten
    I_sp1 = ski.util.random_noise(I, mode="s&p", amount=0.02)
    I_sp2 = ski.util.random_noise(I, mode="s&p", amount=0.10)

    # zurück nach uint8 [0,255]
    def to_u8(X):
        return np.clip(X * 255.0, 0, 255).astype(np.uint8)

    I_g1_u8  = to_u8(I_g1)
    I_g2_u8  = to_u8(I_g2)
    I_sp1_u8 = to_u8(I_sp1)
    I_sp2_u8 = to_u8(I_sp2)

    # anzeige + histogramme
    plt.figure(figsize=(12, 10))

    # original
    plt.subplot(3, 4, 1)
    plt.imshow(I_u8, cmap="gray")
    plt.title("Original")
    plt.axis("off")
    plt.subplot(3, 4, 2)
    plt.hist(I_u8.ravel(), bins="auto")
    plt.title("Hist Original")

    # gaussian var=0.001
    plt.subplot(3, 4, 5)
    plt.imshow(I_g1_u8, cmap="gray")
    plt.title("Gaussian var=0.001")
    plt.axis("off")
    plt.subplot(3, 4, 6)
    plt.hist(I_g1_u8.ravel(), bins=256, range=(0, 255))
    plt.title("Hist Gaussian 0.001")

    # gaussian var=0.01
    plt.subplot(3, 4, 9)
    plt.imshow(I_g2_u8, cmap="gray")
    plt.title("Gaussian var=0.01")
    plt.axis("off")
    plt.subplot(3, 4, 10)
    plt.hist(I_g2_u8.ravel(), bins=256, range=(0, 255))
    plt.title("Hist Gaussian 0.01")

    # salt & pepper amount=0.02
    plt.subplot(3, 4, 3)
    plt.imshow(I_sp1_u8, cmap="gray")
    plt.title("S&P amount=0.02")
    plt.axis("off")
    plt.subplot(3, 4, 4)
    plt.hist(I_sp1_u8.ravel(), bins=256, range=(0, 255))
    plt.title("Hist S&P 0.02")

    # salt & pepper amount=0.10
    plt.subplot(3, 4, 7)
    plt.imshow(I_sp2_u8, cmap="gray")
    plt.title("S&P amount=0.10")
    plt.axis("off")
    plt.subplot(3, 4, 8)
    plt.hist(I_sp2_u8.ravel(), bins=256, range=(0, 255))
    plt.title("Hist S&P 0.10")

    plt.tight_layout()
    plt.show()

#2.6
def a6():
    print("-----------Aufgabe 2.6-----------")

    I = cv2.imread("roentgen_thorax.jpg", cv2.IMREAD_GRAYSCALE)

    if I is None:
        print("Fehler: roentgen_thorax.jpg konnte nicht geladen werden.")
        return

    rows, cols = I.shape
    print("Original shape:", I.shape)

    # viertel der bildgröße = halb so viele zeilen und halb so viele spaltn
    new_rows = rows // 2
    new_cols = cols // 2

    I_small = cv2.resize(I, (new_cols, new_rows))
    print("Verkleinert shape:", I_small.shape)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(I, cmap="gray")
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(I_small, cmap="gray")
    plt.title("1/4 Größe")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

#2.7
def a7():
    print("-----------Aufgabe 2.7-----------")

    I = cv2.imread("roentgen_fuss.jpg", cv2.IMREAD_GRAYSCALE)
    if I is None:
        print("Fehler: roentgen_fuss.jpg konnte nicht geladen werden.")
        return

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(I, cmap="gray")
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.hist(I.ravel(), bins=256, range=(0, 255))
    plt.title("Histogramm")
    plt.tight_layout()
    plt.show()

    thresholds = [60, 100, 140]

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 2, 1)
    plt.imshow(I, cmap="gray")
    plt.title("Original")
    plt.axis("off")

    # Binarisierungen
    for k, t in enumerate(thresholds):
        _, I_bin = cv2.threshold(I, t, 255, cv2.THRESH_BINARY)

        plt.subplot(2, 2, k + 2)
        plt.imshow(I_bin, cmap="gray")
        plt.title(f"THRESH_BINARY, t={t}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

def a8():
    print("-----------Aufgabe 2.8-----------")

    fname = "xr_Laura.dcm"
    dc = pydicom.dcmread(fname)

    print(dc)
    
#2.9
def a9():
    print("-----------Aufgabe 2.9-----------")

    fname = "xr_Laura.dcm"
    ds = pydicom.dcmread(fname)

    I = ds.pixel_array  # Bildmatrix als NumPy-Array

    print("I.shape:", I.shape)
    print("I.dtype:", I.dtype)

    plt.figure(figsize=(6, 6))
    plt.imshow(I, cmap="gray")
    plt.axis("off")
    plt.title("xr_Laura.dcm")
    plt.tight_layout()
    plt.show()
    
    
if __name__ == "__main__":

    print("Aktuelles Arbeitsverzeichnis:", os.getcwd())

    ordnername = "Lektion02" 
    zielpfad = os.path.join(os.getcwd(), ordnername)
    print("Pfad mit os.path.join():", zielpfad)

    # wechseln, falls der ordner existiert damit das skript nicht crasht
    if os.path.exists(zielpfad):
        os.chdir(zielpfad)
        print("Gewechselt nach:", os.getcwd())
    else:
        print("Ordner existiert nicht -> kein os.chdir() durchgeführt.\n")

    #a1()
    #a2()
    #a3()
    #a4()
    #a5()
    #a6()
    #a7()
    #a8()
    a9()
    