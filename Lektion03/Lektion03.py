import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage as ski
    
#2.1
def a1():
    print("-----------Aufgabe 3.1-----------")

#2.2
def a2():
    print("-----------Aufgabe 3.2-----------")

#2.3
def a3():
    print("-----------Aufgabe 3.3-----------")

#2.4
def a4():
    print("-----------Aufgabe 3.4-----------")

#2.5
def a5():
    print("-----------Aufgabe 3.5-----------")

#2.6
def a6():
    print("-----------Aufgabe 3.6-----------")

#2.7
def a7():
    print("-----------Aufgabe 3.7-----------")
    
    
if __name__ == "__main__":

    print("Aktuelles Arbeitsverzeichnis:", os.getcwd())

    ordnername = "Lektion03" 
    zielpfad = os.path.join(os.getcwd(), ordnername)
    print("Pfad mit os.path.join():", zielpfad)

    # wechseln, falls der ordner existiert damit das skript nicht crasht
    if os.path.exists(zielpfad):
        os.chdir(zielpfad)
        print("Gewechselt nach:", os.getcwd())
    else:
        print("Ordner existiert nicht -> kein os.chdir() durchgeführt.\n")

    