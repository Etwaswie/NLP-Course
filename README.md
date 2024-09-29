# NLP-Course
### Задание 1
- Загрузить набор данных по выбору с помощью библиотеки Corus
- Провести релевантную предобработку выбранного датасета:
  - Нормализация
  - Токенизация
  - Удаление стоп-слов
  - Лемматизация/стемминг

 ### Задание 2
- Загрузить набор данных [Spam Or Not Spam](https://www.kaggle.com/datasets/ozlerhakan/spam-or-not-spam-dataset)
- Попробовать и сравнить различные способы векторизации:
  - `sklearn.feature_extraction.text.CountVectorizer`
  - `sklearn.feature_extraction.text.TfidfVectorizer`
- Обучить на полученных векторах модели, с использованием кросс-валидации и подбором гиперпараметров:
  - `sklearn.tree.DecisionTreeClassifier`
  - `sklearn.linear_model.LogisticRegression`
  - Naive Bayes
- Сравнить качество обученных моделей на отложенной выборке

### Задание 3
- Загрузить набор данных [Spam Or Not Spam](https://www.kaggle.com/datasets/ozlerhakan/spam-or-not-spam-dataset) (или любой другой, какой вам нравится)
- Обучить модели и сравнить различные способы векторизации с помощью внутренней оценки (intrinsic):
  - Word2Vec SkipGram / CBOW (параметр sg в `gensim.models.word2vec.Word2Vec`)
  - fastText (можно взять в gensim, или в fasttext как на семинаре)
- Обучить на полученных векторах модели LogisticRegression и сравнить качество на отложенной выборке

### Задание 4
Исходный набор данных - [Fake and real news dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

- Реализовать классификацию двумя моделями: CNN, LSTM
- Сравнить качество обученных моделей

### Задание 5
- Загрузить набор данных Lenta.ru с помощью пакета Corus
- Обучить LDA модель, постараться подобрать адекватные параметры (num_topics, passes, alpha, iterations…)
- Визуализировать результаты работы LDA с помощью pyLDAvis
- Посчитать внутренние метрики обученных моделей LDA (с разными параметрами) и сравнить, соответствует ли метрика визуальному качеству работы моделей
- Обучить модель BigARTM, использовать не менее двух регуляризаторов, оценить качество с помощью метрик
- Реализовать визуализацию топиков BigARTM через pyLDAvis

### Задание 6
- Изучить [главу 3. Fine-tuning a pretrained model туториалов HuggingFace](https://huggingface.co/learn/nlp-course/chapter3/1)
- Выбрать модель архитектуры BERT/GPT в HuggingFace hub для решения задач Text Classification / Text Generation
- Выбрать набор данных для конкретной задачи из HuggingFace Datasets
- Дообучить выбранную модель
- Сравнить качество до и после дообучения с учетом метрик, специфичных для выбранной задачи

### Задание 7
- Изучить [туториал HuggingFace по задаче Question Answering](https://huggingface.co/learn/nlp-course/chapter7/7?fw=pt)
- Выбрать модель в HuggingFace hub для решения задачи Question Answering на русском языке
- Используя набор данных Sberquad дообучить выбранную модель, оценить качество до и после дообучения

### Задание 8
- Выбрать датасет для задачи классификации на HuggingFace
- Обучить "простую" модель классификации - ученика, например, LSTM и модель-учителя (bert или другой трансформер на выбор)
- Обучить дистиллированную простую модель
- Сравнить качество трех моделей: ученика, учителя и дистилированной
