# # Klasyfikacja binarna przelewów bankowych

# ## Autor:
# ##### Patryk Welkier 217409

# Importowanie niezbędnych bibliotek
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict, cross_val_predict, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from matplotlib.colors import LogNorm, Normalize
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier


# # Streszczenie

# Celem projektu jest analiza danych dotyczących transakcji bankowych, w celu wykrycia oszustw oraz ocena skuteczności różnych modeli klasyfikacyjnych. Dane zawierają 284807 transakcji z 30 cechami niezależnymi oraz jedna zależną - Class, która określa nam czy transakcja jest podejrzana - 1, albo niepodejrzana - 0. Zaczołem pracę od analizy danych. W kolejnym kroku przygotowałem dane do użycia modeli, a więc dokonałem podziału na zmienną docelową (Class) od cech (X) oraz podziału zbioru na część uczącą i testową. Mając tak przygotowane dane, zastosowałem modele: Regresja logistyczna, K-Nearest Neighbors (KNN), SGDClassifier oraz przeprowadziłem walidację krzyżową, oraz optymalizację hiperparametrów za pomocą GridSearchCV. Dopełniłem wymienione modele również modelem hybrydowym, wykorzystującym VotingClassifier. Uzyskane wyniki ocenioniłem według takich parametrów jak: dokładność (accuracy), recall i F1-score. Ostateczne zestawienie ukazuje nam te modele posortowane zgodnie z wartością recall, którą uznałem za najważniejszą. Warto zaznaczyć, że uznałem metrykę recall za najważniejszą, ponieważ jako bank, bardziej interesuje nas omyłkowe zablokowanie normalnej transakcji oraz tej podejrzanej, niż przepuszczenie obu i uzyskaniu wyższej dokładności przy tak niezbalansowanym zbiorze danych. Z ostatecznej tabeli wyników odczytujemy, że najlepszą wartość recall uzyskujemy dla modelu hybrydowego KNN + LogisticRegression o wartości 75,5%. F1-score dla tej metody klasyfikacji wynosi 81,3%.

# # Słowa kluczowe
# * Regresja logistyczna
# * K najbliższych sąsiadów (KNN)
# * drzewa decyzyjne
# * F1-score
# * SGDClassifier

# # Przedmiot badania

# ## Opis i analiza danych

# Wśród danych mamy 284807 obserwacji, z czego każda zawiera cechy niezależne takie jak: Time- Czas od pierwszej transakcji w sekundach, Amount- Kwota transakcji oraz V1-V28- wartości wynikające z transformacji PCA (Principal Component Analysis) (te cechy są utajnione) oraz cechę zależną Class, która oznacza, czy transakcja jest oszustwem, czy nie. W zbiorze danych przelewów oznaczonych klasą 1 (czyli oszustwo), mamy 492 (0.17%).
# 
# Atrybuty V1-V28 mają średnie wartości bliskie zeru oraz odchylenia standardowe bliskie jeden, co jest wynikiem przeprowadzenia transformacji PCA. Odchylenia standardowe są stosunkowo podobne, co sugeruje równomierne rozproszenie danych po transformacji. Wartości minimalne i maksymalne są zróżnicowane, co oznacza, że w danych mogą występować wartości odstające.

# Importowanie danych z pliku CSV
data = pd.read_csv('creditcard.csv')
data.head()

# Wyświetlenie liczby wierszy i kolumn w zbiorze danych
data.shape

# Wyświetlenie informacji o danych, w tym typów danych w każdej kolumnie, liczby niepustych wartości i zajmowanej pamięci
data.info()

# ## Analiza liczby wystąpień wartości 0 i 1 w kolumnie 'Class'

# ##### 0 (nie oszustwo): 284,315 przypadków (99.83%)
# ##### 1 (oszustwo): 492 przypadków (0.17%)

# Wyświetlenie liczby wystąpień wartości 0 oraz 1 w kolumnie 'Class'
print(data['Class'].value_counts())

# Obliczenie i wyświetlenie proporcji wartości 0 i 1 w zbiorze danych
print(data['Class'].value_counts()/data.shape[0]*100)

# ## Wizualizacja liczby wystąpień wartości 0 i 1 w kolumnie 'Class'

# Utworzenie wykresu słupkowego, w skali logarytmicznej przedstawiającego liczby wystąpień wartości 0 i 1 w kolumnie 'Class'
sns.countplot(data,x = 'Class')
plt.yscale('log')
plt.show()

# ## Histogramy dla wszystkich kolumn w zbiorze danych

# Utworzenie histogramów dla wszystkich kolumn w zbiorze danych
data.hist(figsize = (20,20))

# ## Macierz korelacji dla wszystkich kolumn liczbowych w zbiorze danych

# Utworzenie macierzy korelacji dla wszystkich kolumn liczbowych w zbiorze danych
sns.heatmap(data.corr(numeric_only= True))

# Generowanie podsumowania statystycznego dla danych numerycznych, takich jak średnia, odchylenie standardowe, kwartyle, min i max
data.describe()

# # Korelacje
# W większości korelacji między atrybutami przeważa wartość 0 albo wartość niewielka, ale warto wyróżnić kilka korelacji.
# 
# Jeśli chodzi o korelacje z klasą, czyli ważna korelacja jeślki chodzi o zrozumienie, które zmienne są przydatne do stwierdzenia, czy transakcja jest podejrzana:
# Najbardziej skorelowany z klasą są
# V4 (0.13) i V2 (0.091). Oznacza to, że są to najważniejsze zmienne do określenia, czy transakcja jest podejrzana. Za to najmniej skorylowane są zmienne V12 (-0.260), V14 (-0.303) i V17 (-0.326), co oznacza, że te zmienne przydadzą się najmniej do stwierdzenia tego, czy transakcja jest podejrzana.
# 
# W przypadku korelacji między zmiennymi wiele z nich ma bardzo niską wartość korelacji (bliskie 0), co sugeruję, że są one niezależne lub słabo skorelowane. Warto wyróżnić kilka największych korelacji:
# V27 i V17 (-0.966) - Wysoka negatywna korelacja.
# V5 i V20 (-0.931) - Wysoka negatywna korelacja.
# V1 i V2 (0.87) - Wysoka pozytywna korelacja.

# ## Sprawdzenie liczby brakujących wartości w każdej kolumnie zbioru danych

# Brak brakujących wartości w zbiorze danych.

# Sprawdzenie liczby brakujących wartości w każdej kolumnie zbioru danych
data.isnull().sum()

# # Tworzenie zbioru testowego
# Ważnym elementem tworzenia modelu statystycznego jest możliwość jego późniejszej ewaluacji. Dlatego niezbędne jest utworzenie zbioru testowego, który będzie miał podobny rozkład danych co zbiór treningowy. Dzięki temu będziemy mogli dokładnie ocenić wydajność naszego modelu, nie pomijając części przypadków. W naszym przypadku, przy liczbie obserwacji wynoszącej ponad


# Tworzenie zmiennych X i Y do podziału danych na zmienne niezależne (X) i zmienną zależną (Y)
X = data.drop('Class', axis = 1)
Y = data['Class']
Y

# Podział danych na zestawy treningowe i testowe, gdzie X_train i Y_train są zestawami treningowymi, a X_test i Y_test są zestawami testowymi
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)
Y_train = np.array(Y_train)
# Wyświetlenie liczby elementów w tablicy Y_train
Y_train.shape

# Wyświetlenie liczby elementów w tablicy X_train. Liczba 227845 to ilość obserwacji w zbiorze treningowym
X_train.shape

# # Transformacje danych

# ## Skalowanie

# Używając StandardScaler, transformujemy dane tak, aby odchylenie standardowe wynosiło 1.

# Utworzenie obiektu scaler klasy StandardScaler do standaryzacji danych
scaler = StandardScaler()
# Standaryzacja danych treningowych 
X_scaled_train = scaler.fit_transform(X_train)
X_scaled_test = scaler.fit_transform(X_test)
X_scaled_train

# Wyświetlenie przeskalowanych dane treningowych
X_scaled_train_pd = pd.DataFrame(X_scaled_train)
X_scaled_train_pd

# Generowanie podsumowania statystycznego dla przeskalowanych danych treningowych
X_scaled_train_pd.describe()

# # Funkcja służąca do oceny modeli przez walidację krzyżową  

# * Accuracy: To procentowa wartość określająca stosunek liczby poprawnie sklasyfikowanych przypadków do całkowitej liczby przypadków. Jest to ogólna metryka używana do oceny jakości klasyfikatora, ale może być myląca w przypadku niezrównoważonych zbiorów danych.
# 
# * Recall: Jest to miara zdolności modelu do poprawnej identyfikacji wszystkich istotnych przypadków w zestawie danych. To stosunek liczby prawdziwie pozytywnych przypadków do sumy prawdziwie pozytywnych i fałszywie negatywnych przypadków. Jest szczególnie przydatna w przypadkach, gdzie brakujące prawdziwie pozytywne przypadki mogą być kosztowne lub niebezpieczne.
# 
# * F1-score: Jest to średnia harmonicznego precision i recall. Jest używana do równoważenia precision i recall, ponieważ czasami zależy nam na minimalizacji zarówno fałszywie pozytywnych, jak i fałszywie negatywnych wyników. F1-score osiąga swoją najlepszą wartość przy 1 (idealnym precision i recall) i najgorszą wartość przy 0.
# 
# Te metryki są używane w analizie danych do oceny skuteczności modeli klasyfikacji i pomagają zrozumieć, jak dobrze model radzi sobie z przewidywaniem wyników. Ważne jest dostosowanie wyboru metryki do konkretnego problemu i kontekstu biznesowego.

# ![image.png](attachment:e59a2e43-3a32-4935-8af3-01244e7f520e.png)

# Definicja funkcji evaluate_model, która ocenia wybrany model za pomocą walidacji krzyżowej
# model - model do oceny
# X - cechy niezależne
# Y - cechy zależne
# n_splits - liczba podziałów w walidacji krzyżowej (domyślnie 5)
# accF - funkcja do obliczenia dokładności (domyślnie accuracy_score)
# recF - funkcja do obliczenia recall (domyślnie recall_score)
def evaluate_model(model, X, Y, n_splits=5, accF=accuracy_score, recF=recall_score):
    # Inicjalizacja obiektu StratifiedKFold do walidacji krzyżowej
    skf = StratifiedKFold(n_splits=n_splits)
    acc = 0  # Inicjalizacja zmiennej przechowującej sumę dokładności dla wszystkich podziałów
    rec = 0  # Inicjalizacja zmiennej przechowującej sumę miary odzysku dla wszystkich podziałów
    f1 = 0   # Inicjalizacja zmiennej przechowującej sumę F1-score dla wszystkich podziałów
    
    # Pętla po podziałach danych w walidacji krzyżowej
    for train_index, test_index in skf.split(X, Y):
        x_train, x_test = X[train_index], X[test_index]  # Podział danych na zbiory treningowy i testowy w walidacji krzyżowej
        y_train, y_test = Y[train_index], Y[test_index]  # Podział etykiet klas na zbiory treningowy i testowy w walidacji krzyżowej

        model.fit(x_train, y_train)
        
        # Przewidywanie etykiet klas dla danych testowych
        y_pred = model.predict(x_test)
        
        # Obliczenie dokładności i miary odzysku dla bieżącego podziału
        acc_current = accF(y_test, y_pred)
        acc += acc_current
        rec_current = recF(y_test, y_pred)
        rec += rec_current
        
        # Obliczenie F1-score dla bieżącego podziału
        f1 += f1_score(y_test, y_pred)
    
    # Obliczenie średniej dokładności, miary odzysku i F1-score ze wszystkich podziałów walidacji krzyżowej
    acc /= n_splits
    rec /= n_splits
    f1 /= n_splits

    # Wyświetlenie wyników oceny modelu
    print('Dokładność: ', acc)
    print('Recall: ', rec)
    print('F1-score: ', f1)
    
    # Zwrócenie dokładności, miary odzysku i F1-score
    return acc, rec, f1



def final_eval(model):

    #Wyliczenie metryk
    y_pred = model.predict(X_scaled_test)
    acc = accuracy_score(Y_test, y_pred)
    rec = recall_score(Y_test, y_pred)
    f1 = f1_score(Y_test, y_pred)

    # Wyświetlenie confusion matrix
    cf_matrix = confusion_matrix(Y_train, y_sgd_pred)
    sns.heatmap(cf_matrix, square=True, norm=LogNorm())
    
    # Wyświetlenie wyników oceny modelu
    print('Precyzja: ', acc)
    print('Recall: ', rec)
    print('F1-score: ', f1)

# # 1. Logistic Regression

# Regresja logistyczna – jedna z metod regresji używanych w statystyce w przypadku, gdy zmienna zależna jest na skali dychotomicznej (przyjmuje tylko dwie wartości). Zmienne niezależne w analizie regresji logistycznej mogą przyjmować charakter nominalny, porządkowy, przedziałowy lub ilorazowy. W przypadku zmiennych nominalnych oraz porządkowych następuje ich przekodowanie w liczbę zmiennych zero-jedynkowych taką samą lub o 1 mniejszą niż liczba kategorii w jej definicji.
# 
# Zwykle wartości zmiennej objaśnianej wskazują na wystąpienie lub brak wystąpienia pewnego zdarzenia, które chcemy prognozować. Regresja logistyczna pozwala wówczas na obliczanie prawdopodobieństwa tego zdarzenia (tzw. prawdopodobieństwo sukcesu).
# 
# Formalnie model regresji logistycznej jest uogólnionym modelem liniowym (GLM), w którym użyto logitu jako funkcji wiążącej.


# Utworzenie obiektu logistic_model klasy LogisticRegression do modelowania regresji logistycznej
logistic_model = LogisticRegression()

# Wywołanie funkcji evaluate_model do oceny modelu regresji logistycznej
# X_scaled_train - przeskalowane cechy danych treningowych
# Y_train - etykiety klas treningowych
# acc_LogisticRegression, rec_LogisticRegression, f1_LogisticRegression - wyniki oceny modelu (dokładność, recall, F1-score)
acc_LogisticRegression, rec_LogisticRegression, f1_LogisticRegression = evaluate_model(logistic_model, X_scaled_train, Y_train)
    
# # 2. SGD Classifier

# SGD Classifier (Stochastic Gradient Descent Classifier) jest modelem uczenia maszynowego, który należy do rodziny klasyfikatorów liniowych. Jest to elastyczny i wydajny klasyfikator, który dobrze radzi sobie z dużymi zestawami danych, co czyni go idealnym wyborem dla projektów z analizy danych.
# 
# Główne cechy modelu SGD Classifier to:
# 
# * Efektywność obliczeniowa: Wykorzystuje metodę optymalizacji stochastycznego spadku gradientu, co umożliwia szybkie dostosowywanie modelu do dużych zbiorów danych. Dzięki temu jest wydajny nawet w przypadku bardzo dużych zbiorów danych.
# 
# * Wielozadaniowość: Może być stosowany zarówno do problemów klasyfikacji binarnej, jak i wieloklasowej. Jest to bardzo elastyczny klasyfikator, który można dostosować do różnych scenariuszy biznesowych.
# 
# * Regularyzacja: Posiada wbudowane opcje regularyzacji, co pomaga w zapobieganiu przeuczeniu i poprawia ogólną zdolność generalizacji modelu.
# 
# * Parametryzacja: Dzięki różnym parametrom, takim jak liczba iteracji, stała uczenia czy funkcja straty, model można dostosować do konkretnej analizy danych i problemu klasyfikacji.
# 
# SGD Classifier jest używany w projekcie z metod analizy danych ze względu na swoją wszechstronność, wydajność i możliwość dostosowania do różnorodnych zadań klasyfikacji. Jego zastosowanie może prowadzić do skutecznych i skalowalnych rozwiązań w analizie danych.

# # Confusion Matrix dla SGD Classifier
# 

# Macierz pomyłek (confusion matrix) to tabela używana w analizie klasyfikacji, która pokazuje liczbę poprawnych i błędnych klasyfikacji wykonanych przez model predykcyjny na zestawie danych testowych. Składa się z czterech komórek, które reprezentują liczbę prawdziwie pozytywnych (True Positive), fałszywie pozytywnych (False Positive), prawdziwie negatywnych (True Negative) i fałszywie negatywnych (False Negative). Ta struktura pozwala na ocenę wydajności modelu klasyfikacji.

# Utworzenie obiektu sgd_clf klasy SGDClassifier do klasyfikacji za pomocą metody SGD
# max_iter=1000 - maksymalna liczba iteracji optymalizacji
# tol=1e-3 - kryterium zatrzymania optymalizacji
# random_state=42 - ustawienie wartości random_state dla powtarzalności wyników
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)

# Wykonanie predykcji za pomocą walidacji krzyżowej (cv=3) i zwrócenie predykcji dla każdej próbki
y_sgd_pred = cross_val_predict(sgd_clf, X_scaled_train, Y_train, cv = 3)

# Obliczenie macierzy pomyłek dla prawdziwych etykiet klas (Y_train) i przewidywanych etykiet klas (y_sgd_pred)
cf_matrix = confusion_matrix(Y_train, y_sgd_pred)
sns.heatmap(cf_matrix, square=True, norm=LogNorm())

# Utworzenie obiektu model klasy SGDClassifier do klasyfikacji za pomocą metody SGD
# max_iter=1000 - maksymalna liczba iteracji optymalizacji
# tol=1e-3 - kryterium zatrzymania optymalizacji
# random_state=42 - ustawienie wartości random_state dla powtarzalności wyników
model = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)

# Wywołanie funkcji evaluate_model do oceny modelu klasyfikatora SGD
# X_scaled_train - przeskalowane cechy danych treningowych
# Y_train - etykiety klas treningowych
# acc_SGDClassifier, rec_SGDClassifier, f1_SGDClassifier - wyniki oceny modelu (dokładność, recall, F1-score)
acc_SGDClassifier, rec_SGDClassifier, f1_SGDClassifier = evaluate_model(model, X_scaled_train, Y_train)

# # 3. K-Nearest Neighbors (KNN)

# Algorytm k najbliższych sąsiadów (lub algorytm k-nn z ang. k nearest neighbours) – jeden z algorytmów regresji nieparametrycznej używanych w statystyce do prognozowania wartości pewnej zmiennej losowej. Może również być używany do klasyfikacji.
# 
# ## Założenia:
# 
# * Dany jest zbiór uczący zawierający obserwacje, z których każda ma przypisany wektor zmiennych objaśniających
# oraz wartość zmiennej objaśnianej.
# 
# * Dana jest obserwacja z przypisanym wektorem zmiennych objaśniających, dla której chcemy prognozować wartość zmiennej objaśnianej.
# 
# ## Algorytm polega na:
# 
# * porównaniu wartości zmiennych objaśniających dla obserwacji z wartościami tych zmiennych dla każdej obserwacji w zbiorze uczącym.
# 
# * wyborze (ustalona z góry liczba) najbliższych do obserwacji ze zbioru uczącego.
# 
# * uśrednieniu wartości zmiennej objaśnianej dla wybranych obserwacji, w wyniku czego uzyskujemy prognozę.
# 
# ![image.png](attachment:6daf1cdd-9aba-4454-b986-8727a95f4edd.png)


# Ustawienie liczby sąsiadów na 5
k = 5 

# Utworzenie obiektu model klasy KNeighborsClassifier do klasyfikacji za pomocą metody k-najbliższych sąsiadów
model = KNeighborsClassifier(n_neighbors=k)

# Wywołanie funkcji evaluate_model do oceny modelu klasyfikatora k-najbliższych sąsiadów
# X_scaled_train - przeskalowane cechy danych treningowych
# Y_train - etykiety klas treningowych
# acc_KNN, rec_KNN, f1_KNN - wyniki oceny modelu (dokładność, recall, F1-score)
acc_KNN, rec_KNN, f1_KNN = evaluate_model(model, X_scaled_train, Y_train)

# # GridSearchCV

# GridSearchCV to technika optymalizacji hiperparametrów, która polega na przeszukiwaniu przestrzeni hiperparametrów w celu znalezienia najlepszej kombinacji dla danego modelu. Jest to potężne narzędzie w analizie danych, które pomaga w wyborze optymalnych parametrów modelu, co prowadzi do poprawy jego wydajności i skuteczności.
# 
# Główne cechy techniki GridSearchCV to:
# 
# * Przeszukiwanie siatki: GridSearchCV przeszukuje określoną siatkę wartości parametrów, które zdefiniowano wcześniej. Możliwe wartości parametrów są określone przez użytkownika i obejmują różne kombinacje, które chcemy zbadać.
# 
# * Walidacja krzyżowa: GridSearchCV używa techniki walidacji krzyżowej do oceny skuteczności każdej kombinacji hiperparametrów. Jest to istotne, aby uniknąć nadmiernej optymalizacji (overfittingu) do konkretnego zestawu danych.
# 
# * Optymalizacja hiperparametrów: Celem GridSearchCV jest znalezienie takich wartości hiperparametrów, które maksymalizują wybraną metrykę oceny modelu, w naszym przypadku Recall.
# 
# * Dostrojenie modelu: GridSearchCV umożliwia dostrojenie różnych modeli, w tym LogisticRegression, SGD Classifier czy KNN, co pozwala na uzyskanie optymalnych wyników w zależności od charakteru danych i problemu.
# 
# * GridSearchCV jest używany w projekcie do optymalizacji hiperparametrów modeli, co pomaga w znalezieniu najlepszych kombinacji parametrów i poprawia skuteczność klasyfikacji. Jest to istotny krok w procesie tworzenia modeli w analizie danych, który może znacząco wpłynąć na wyniki predykcyjne i jakość rozwiązania.

# ## 4. Logistic Regression with GridSearchCV

# Utworzenie obiektu grid_LG klasy GridSearchCV do przeszukiwania siatki hiperparametrów dla modelu regresji logistycznej
# estimator=LogisticRegression(max_iter=1000) - model bazowy, w tym przypadku regresja logistyczna z maksymalną liczbą iteracji 1000
# param_grid={'class_weight': [{0: 1, 1: v} for v in range(1, 4)]} - siatka hiperparametrów, w tym wagi klas dla klas 0 i 1
# scoring='recall' - wybór miary do oceny modelu, w tym przypadku recall (miara odzysku)
# cv=2 - liczba podziałów w walidacji krzyżowej
# verbose=2 - poziom szczegółowości wyjścia podczas dopasowywania modelu
# n_jobs=-1 - liczba zadań do równoległego wykonywania, -1 oznacza wykorzystanie wszystkich dostępnych rdzeni CPU
grid_LG = GridSearchCV(
    estimator=LogisticRegression(max_iter=1000), 
    param_grid={'class_weight': [{0: 1, 1: v} for v in range(1, 4)]}, 
    scoring='recall', 
    cv=2, 
    verbose=2,  
    n_jobs=-1
)

# Dopasowanie siatki hiperparametrów do danych treningowych i wybór najlepszego estymatora
model_gridLG = grid_LG.fit(X_scaled_train, Y_train).best_estimator_

# Ocena najlepszego estymatora za pomocą funkcji evaluate_model
# acc_gridLogisticRegression, rec_gridLogisticRegression, f1_gridLogisticRegression - wyniki oceny modelu (dokładność, miara odzysku, F1-score)
acc_gridLogisticRegression, rec_gridLogisticRegression, f1_gridLogisticRegression = evaluate_model(model_gridLG, X_scaled_train, Y_train)


# ## 5. SGD Classifier with GridSearchCV

# Utworzenie obiektu grid_SGDC klasy GridSearchCV do przeszukiwania siatki hiperparametrów dla modelu klasyfikatora SGD
# estimator=SGDClassifier(max_iter=1000, tol=1e-3, random_state=42) - model bazowy, klasyfikator SGD z maksymalną liczbą iteracji 1000, tolerancją 1e-3 i stanem losowym 42
# param_grid - siatka hiperparametrów do przeszukania, w tym współczynniki alpha, funkcje straty i kary
# scoring='recall' - wybór miary do oceny modelu, w tym przypadku recall (miara odzysku)
# cv=2 - liczba podziałów w walidacji krzyżowej
# verbose=2 - poziom szczegółowości wyjścia podczas dopasowywania modelu
# n_jobs=-1 - liczba zadań do równoległego wykonywania, -1 oznacza wykorzystanie wszystkich dostępnych rdzeni CPU
grid_SGDC = GridSearchCV(
    estimator=SGDClassifier(max_iter=1000, tol=1e-3, random_state=42),
    param_grid={
        'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
        'loss': ['hinge', 'log_loss'],
        'penalty': ['l2', 'elasticnet']
    },
    scoring='recall',
    cv=2,
    verbose=2,
    n_jobs=-1
)

# Dopasowanie siatki hiperparametrów do danych treningowych i wybór najlepszego estymatora
best_estimator_SGD = grid_SGDC.fit(X_scaled_train, Y_train).best_estimator_

# Ocena uzyskanego estymatora za pomocą funkcji evaluate_model
# acc_gridSGDClassifier, rec_gridSGDClassifier, f1_gridSGDClassifier - wyniki oceny modelu (dokładność, miara odzysku, F1-score)
acc_gridSGDClassifier, rec_gridSGDClassifier, f1_gridSGDClassifier = evaluate_model(best_estimator_SGD, X_scaled_train, Y_train)

# Wyświetlenie najlepszych parametrów uzyskanych w trakcie przeszukiwania siatki hiperparametrów
grid_SGDC.best_params_

# ## 6. K-Nearest Neighbors with GridSearchCV

# Utworzenie obiektu grid_KNN klasy GridSearchCV do przeszukiwania siatki hiperparametrów dla modelu klasyfikatora k-najbliższych sąsiadów
# estimator=KNeighborsClassifier(n_neighbors=k) - model bazowy, klasyfikator k-najbliższych sąsiadów z ustaloną liczbą sąsiadów k
# param_grid - siatka hiperparametrów do przeszukania, w tym liczba sąsiadów
# scoring='recall' - wybór miary do oceny modelu, w tym przypadku recall (miara odzysku)
# cv=2 - liczba podziałów w walidacji krzyżowej
# verbose=2 - poziom szczegółowości wyjścia podczas dopasowywania modelu
# n_jobs=-1 - liczba zadań do równoległego wykonywania, -1 oznacza wykorzystanie wszystkich dostępnych rdzeni CPU
grid_KNN = GridSearchCV(
    estimator=KNeighborsClassifier(n_neighbors=k),
    param_grid={'n_neighbors': [3, 5, 8, 15, 25, 50]},
    scoring='recall',
    cv=2,
    verbose=2,
    n_jobs=-1
)

# Dopasowanie siatki hiperparametrów do danych treningowych i wybór najlepszego estymatora
model_gridKNN = grid_KNN.fit(X_scaled_train, Y_train).best_estimator_

# Ocena uzyskanego estymatora za pomocą funkcji evaluate_model
# acc_gridKNN, rec_gridKNN, f1_gridKNN - wyniki oceny modelu (dokładność, miara odzysku, F1-score)
acc_gridKNN, rec_gridKNN, f1_gridKNN = evaluate_model(model_gridKNN, X_scaled_train, Y_train)

# Wyświetlenie najlepszych parametrów uzyskanych w trakcie przeszukiwania siatki hiperparametrów
grid_KNN.best_params_

# # 7. Hybrid Model (Voting Classifier)

# Hybrydowy model, znany także jako Voting Classifier, to technika łączenia wielu modeli w celu poprawy skuteczności klasyfikacji. W przypadku tego projektu łączymy regresję logistyczną i KNN (K-Nearest Neighbors) w ramach jednego modelu, aby wykorzystać moc obu podejść.
# 
# Główne cechy hybrydowego modelu (Voting Classifier) to:
# 
# * Łączenie różnych podejść: Model ten łączy regresję logistyczną i KNN, wykorzystując ich różne cechy i siły w celu uzyskania bardziej zrównoważonej i wszechstronnej skuteczności klasyfikacji.
# 
# * Zespołowa decyzja: W hybrydowym modelu, każdy z modeli (regresja logistyczna i KNN) oddzielnym głosuje na predykcje. Ostateczna decyzja klasyfikacyjna jest podejmowana poprzez głosowanie większościowe lub średnią wagową predykcji wszystkich modeli.
# 
# * Zrównoważona skuteczność: Poprzez łączenie różnych modeli, hybrydowy model może osiągnąć bardziej zrównoważoną skuteczność.
# 
# * Redukcja ryzyka: W przypadku, gdy jeden z modeli zawodzi lub ma słabą wydajność w określonych przypadkach, hybrydowy model może zredukować ryzyko poprzez uwzględnienie predykcji innych modeli.
# 
# Hybrydowy model (Voting Classifier), łączący regresję logistyczną i KNN, jest używany w projekcie jako podejście do zwiększenia skuteczności klasyfikacji poprzez wykorzystanie różnych metod. Jego elastyczność i zdolność do zrównoważonej oceny przypadków czynią go atrakcyjnym narzędziem w analizie danych.


# Utworzenie obiektu hybrid_model klasy VotingClassifier, który agreguje modele model_gridLG i model_gridKNN
# estimators - lista estymatorów, w tym modeli regresji logistycznej i klasyfikatora k-najbliższych sąsiadów
# voting='soft' - tryb głosowania, głosowanie miękkie
hybrid_model = VotingClassifier(
    estimators=[
        ('LG', model_gridLG), 
        ('KNN', model_gridKNN)
    ], 
    voting='soft'
)

# Ocena hybrydowego modelu za pomocą funkcji evaluate_model
# acc_hybrid, rec_hybrid, f1_hybrid - wyniki oceny modelu (dokładność, miara odzysku, F1-score)
acc_hybrid, rec_hybrid, f1_hybrid = evaluate_model(hybrid_model, X_scaled_train, Y_train)

# # Rezultaty 

# Utworzenie słownika results zawierającego wyniki oceny różnych modeli
results = {
    'Model': ['Logistic Regression', 'SGD Classifier', 'K-Nearest Neighbors', 'Logistic Regression with GridSearchCV', 'SGD Classifier with GridSearchCV', 'K-Nearest Neighbors with GridSearchCV', 'Hybrid KNN+LG'],
    'Accuracy': [acc_LogisticRegression, acc_SGDClassifier, acc_KNN, acc_gridLogisticRegression, acc_gridSGDClassifier, acc_gridKNN, acc_hybrid],
    'Recall': [rec_LogisticRegression, rec_SGDClassifier, rec_KNN, rec_gridLogisticRegression, rec_gridSGDClassifier, rec_gridKNN, rec_hybrid],
    'F1 Score': [f1_LogisticRegression, f1_SGDClassifier, f1_KNN, f1_gridLogisticRegression, f1_gridSGDClassifier, f1_gridKNN, f1_hybrid ]
}

# Utworzenie obiektu DataFrame df_results na podstawie słownika results
df_results = pd.DataFrame(results)

# Ustawienie kolumny 'Model' jako indeksu
df_results.set_index('Model', inplace=True)


# Na końcu sortujemy modele po metryce Recall, ponieważ jest to procent wykrytych anomalii podejrzanych przelewów.

# Utworzenie nowego DataFrame df_sorted poprzez posortowanie DataFrame df_results według kolumny 'Recall' w kolejności malejącej
df_sorted = df_results.sort_values(by=['Recall'], ascending=False)

# Wyświetlenie posortowanego DataFrame df_sorted
df_sorted

# # Ewaluacja modelu hybrydowego na zbiorze testowym i Confusion Matrix

final_eval(hybrid_model)

# # Podsumowanie
# 
# Z uzyskanych ostatecznych metryk na zbiorze treningowym dla poszczególnych modeli wnisokuje, że najlepszy okazał się model hybrydowy łączący zoptymalizowanę KNN + LogisticRegression. Recall, na którym najbardziej nam zależało, jest na poziomie 77,5%, a F1-score wynosi 82%. Przeprowadzając ewaluację tego modelu na zbiorze testowym, zauważyłem marginalny spadek metryk: recall'u o 2 p.p. i F1-score o niecały 1 p.p. Jest to normalne dla każdego modelu, ponieważ generalizacja nowych przypadków zawsze stanowi problem.

# # Bibliografia
# <ol>
#     <li>Emin Aleskerov, Bernd Freisleben, and Bharat Rao. Cardwatch: a neural network based database mining system for credit card fraud detection. In Proceedings of the IEEE/IAFE 1997 computational intelligence for financial engineering (CIFEr), 220–226. IEEE, 1997.</li>
#     <li>Leo Breiman. Random forests. Machine learning, 45(1):5–32, 2001</li>
#     <li>Fabrizio Carcillo. Beyond Supervised Learning in Credit Card Fraud Detection: A Dive into Semi-supervised and Distributed Learning. Université libre de Bruxelles, 2018.</li>
#     <li>Andrea Dal Pozzolo. Adaptive machine learning for credit card fraud detection. Université libre de Bruxelles, 2015.</li>
#     <li>Alejandro Correa Bahnsen, Djamila Aouada, Aleksandar Stojanovic, and Björn Ottersten. Feature engineering strategies for credit card fraud detection. Expert Systems with Applications, 51:134–142, 2016.</li>
#     <li>Pan, S. S. and Zhang, W. J. 2017. Fraudulent detection based on hybrid approach. Journal of East China Normal University(Natural Science). 2017(05):125--137.</li>
# </ol>
