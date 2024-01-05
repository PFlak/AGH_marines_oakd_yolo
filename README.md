# Projekt uczenia robota podwodnego

## Ważne podstawowe informacje

Programu sa stworzone by współpracować z strukturą bazy YOLOv8
Czyli musi istnieć folder datasets o odpowiedniej budowie oraz plik data.yaml w katalogu roboczym

Dodatkowo by upewnic sie ze yolo używa naszej bazy danych w pliku (C:
\Users\n2one\AppData\Roaming\Ultralytics\settings.yaml) usiń path

## Budowa formatu bazy yolov8
Folder datasets zawiera 3 foldery(test, train, valid)

Każdy z nich ma strukture 2 folderów
- images zawierajacy zdjecia
- labels zawierajacy oznaczenia zdjecia
  - Posiadają tą samą nazwe co zdjecie które labelują z rozszerzeniem txt
  - struktura pliku to w każdej lini jest jedno zaznaczenie pierwsza liczba to numer elementu( co przedstawia zaznaczenie) następne 2 to są pozycje środka prostokąta jako znormalizowane wartości następne 2 liczby to szerokość i wysokość prostokąta w wersji znormalizowanej


## Opis programów

### train.py

Program do trenowania sieci w przypadku pierwszego uruchomienia używwa on sieci yolov8n
w póżniej sieci z najwyższym numerem z forderu z wersjami modelów.

### WAŻNE

w przypadku rozpoczęcia i nie zakończenia uczenia może być potrzeba usuniecia najnowszej wersji modelu

Parametry programu

--model_name jak nazywane będą foldery poszczególnych uczeń (default version)

--out_path nazwa folderu w którym występują wersje modeli i bedą tam zapisywane (default versions)

--epochs liczka epok uczenia(default 10)

--gpu_id numer karty graficznej(0,1,2,...), w przypadku checi obliczen na procesorze podaj wartość cpu (default 0)

### example_use.py
Przykładowe użycie sieci neuronowej z frameworkiem opencv korzysta z najnowszej wersji z folderu versions
Po uruchomieniu zaznacz na zdjęciach z folderu test w bazie i wynik tego zwraca do folderu out

Parametry wewnąrz programu

x_line minimalna szerokość zaznaczenia by zostało ono narysowane (default 10)

conf_limit minimalna pewność by zostało narysowane zaznaczenie (default 0.5) 

input_dir folder z którego wybieramy zdjecia do zaznaczania (default ./datasets/test)

out_dir = folder do którego są zwracane wyniki (default ./out/)

### possion_dataset.py
Program wykorzystujący algorytm poisson-image-editing 

Parametry programu

--background_dir folder zawierający tła (default backgrounds)

--input_dir folder z zdjeciami do algorytmu i folder do którego będą zapisywane (default datasets/train)

--pool_size liczba wątków (default 16)
