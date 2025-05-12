# OpenCL Példaprojektek

Ez a repository négy különböző OpenCL-alapú programot tartalmaz, amelyek különféle számítási feladatokat hajtanak végre párhuzamosan.

## Projektek

### 1. `vektorok`
Egyszerű vektoros műveleteket hajt végre OpenCL segítségével.

### 2. `huffman`
Huffman-kódol szöveget. A szöveg lehet előre megadott vagy akár random generált is.

### 3. `matrixok`
Mátrixműveleteket valósít meg párhuzamosan. A mátrixok mérete állítható.

### 4. `randomsort`
A bogosort, másnéven stupid sort algoritmust valósítja meg párhuzamosítással. Ez egy rendkívül nem hatékony rendezési algoritmus, mely úgy működik, hogy véletlenszerűen cserélgeti a tömb elemeit addig, míg az rendezve nincs. Párhuzamosításnál, az összes szál saját tömbbel dolgozik az adatvesztés elkerülése érdekében. Amint a tömböt sikerült rendeznie egy szálnak, leáll a többi szál is.
