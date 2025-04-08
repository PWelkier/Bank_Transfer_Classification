# 🏦 Bank Transfer Classification

## 🎯 Cel projektu

Celem projektu jest analiza danych dotyczących transakcji bankowych w celu **wykrycia oszustw (fraud detection)** oraz **ocena skuteczności różnych modeli klasyfikacyjnych**.

Dane zawierają **284 807 transakcji**, z czego każda opisana jest przez **30 cech niezależnych** oraz jedną cechę zależną – `Class`, która określa, czy transakcja jest:
- `1` – podejrzana (oszustwo),
- `0` – niepodejrzana (normalna).

## 🧪 Etapy projektu

1. **Analiza danych (EDA)** – wstępne zrozumienie rozkładu i właściwości danych.
2. **Przygotowanie danych**:
   - Oddzielenie zmiennej docelowej (`Class`) od cech (`X`),
   - Podział na zbiór treningowy i testowy.
3. **Trenowanie modeli klasyfikacyjnych**:
   - Logistic Regression,
   - K-Nearest Neighbors (KNN),
   - SGDClassifier,
   - VotingClassifier (model hybrydowy).
4. **Walidacja i optymalizacja**:
   - Walidacja krzyżowa,
   - GridSearchCV do dostrajania hiperparametrów.
5. **Ewaluacja modeli**:
   - Metryki: `accuracy`, `recall`, `f1-score`,
   - Ranking modeli według wartości **recall** – najważniejszej metryki w tym kontekście.

## 🤖 Dlaczego recall?

W przypadku wykrywania oszustw bankowych **recall (czułość)** jest najistotniejszy, ponieważ:
- Wolimy **omyłkowo zablokować normalną transakcję** niż **przepuścić fałszywą**,
- Zbiór danych jest silnie **niezbalansowany** – większość to transakcje niepodejrzane.

## 🏁 Wyniki końcowe

Najlepsze wyniki osiągnął **model hybrydowy** – `VotingClassifier` złożony z:
- `KNN`,
- `LogisticRegression`.

📊 **Wyniki dla najlepszego modelu**:
- `Recall`: **75,5%**
- `F1-score`: **81,3%**

## 🧰 Technologie i biblioteki

- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib / Seaborn (do EDA)
- Jupyter Notebook
