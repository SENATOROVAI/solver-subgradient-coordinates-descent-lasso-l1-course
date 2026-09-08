# MAE, Lasso и ElasticNet на Python — субградиентный и координатный спуск

Практический репозиторий к курсу Stepik **«MAE, Lasso и ElasticNet: практический курс»**. Здесь собраны воспроизводимые примеры на Python: линейная регрессия с функциями потерь MAE и MSE, L1/L2-регуляризация, Lasso, Ridge, ElasticNet, soft-thresholding, субградиентный метод и координатный спуск.

Материалы подходят для самостоятельного изучения машинного обучения, подготовки к занятиям по оптимизации и разбора того, как устроены линейные модели до вызова готового API scikit-learn.

## Что есть в репозитории

| Раздел | Содержание |
| --- | --- |
| [`python/pure_core.py`](python/pure_core.py) | Основной модуль: прогнозы, остатки, MAE/MSE, масштабирование, Lasso и ElasticNet coordinate descent. |
| [`python/snippets/`](python/snippets/) | 20 полных примеров: по одному для каждой главы курса. |
| [`python/listings/`](python/listings/) | Короткие листинги алгоритмов: медиана, soft-thresholding, функция цели, обновление координаты и другие. |
| [`python/exercise_solutions/`](python/exercise_solutions/) | Решения практических заданий глав 01–20. |
| [`python/extensions/`](python/extensions/) | 26 дополнительных заданий по оптимизации, статистике и геометрии регуляризации. |
| [`python/extension_solutions/`](python/extension_solutions/) | Решения дополнительных заданий. |
| [`python/stepik_solutions/`](python/stepik_solutions/) | Решения Python-задач Stepik: `009_code` и `014_code` каждой главы. |
| [`module_2.1/task1_1.py`](module_2.1/task1_1.py) | Самостоятельная задача: медиана и минимальная средняя абсолютная ошибка. |

## Темы и ключевые слова

`Python`, `machine learning`, `линейная регрессия`, `MAE`, `MSE`, `L1-регуляризация`, `L2-регуляризация`, `Lasso`, `Ridge`, `ElasticNet`, `coordinate descent`, `координатный спуск`, `subgradient descent`, `субградиентный метод`, `soft thresholding`, `proximal gradient`, `ISTA`, `KKT conditions`, `оптимизация`, `scikit-learn`.

## Программа курса

Код последовательно проходит путь от простых вычислений к оптимизации регуляризованных моделей:

1. прогнозы, остатки, абсолютная и квадратичная ошибки;
2. медиана и MAE, среднее и MSE, перебор параметров;
3. субградиентный метод для MAE;
4. масштабирование признаков и интерпретация коэффициентов;
5. L1-, L2- и ElasticNet-регуляризация;
6. soft-thresholding и точные координатные обновления;
7. координатный спуск для Lasso и ElasticNet;
8. критерии оптимальности, KKT-проверки, устойчивость коэффициентов и сравнение со scikit-learn.

Для запуска конкретной главы используйте файлы [`python/snippets/ch01.py`](python/snippets/ch01.py) … [`python/snippets/ch20.py`](python/snippets/ch20.py).

## Быстрый старт

Требуется Python 3.10 или новее.

```bash
git clone https://github.com/SENATOROVAI/solver-subgradient-coordinates-descent-lasso-l1-course.git
cd solver-subgradient-coordinates-descent-lasso-l1-course
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
cd python
python run_chapter.py 7
```

Запустить все основные примеры:

```bash
cd python
python run_all.py
```

Запустить решения упражнений и расширений:

```bash
python run_exercise_solutions.py
python run_extensions.py
python run_extension_solutions.py
```

## Проверка кода

После установки зависимостей выполните из каталога `python/`:

```bash
python -m unittest test_core.py test_optim_extra.py test_qp_reference.py
```

Тесты сверяют собственные реализации оптимизации с численными эталонами и граничными случаями.

## Решения задач Stepik

Полные ответы вынесены из стартовых шаблонов заданий, чтобы их можно было открыть только при необходимости. Для главы `NN` решения находятся здесь:

```text
python/stepik_solutions/lesson_NN/009.py
python/stepik_solutions/lesson_NN/014.py
```

Например, решения главы 01: [`009.py`](python/stepik_solutions/lesson_01/009.py) и [`014.py`](python/stepik_solutions/lesson_01/014.py).

## Используемые библиотеки

- [NumPy](https://numpy.org/) — массивы и численные проверки;
- [Matplotlib](https://matplotlib.org/) — графики;
- [SciPy](https://scipy.org/) — вспомогательные численные процедуры;
- [scikit-learn](https://scikit-learn.org/) — сравнение с промышленными реализациями Lasso и ElasticNet.

## Навигация для Stepik

Соответствие статей и файлов простое: **глава 01 → `python/snippets/ch01.py`**, …, **глава 20 → `python/snippets/ch20.py`**. Для готовых ссылок на примеры и решения используйте [каталог решений Stepik](https://github.com/SENATOROVAI/solver-subgradient-coordinates-descent-lasso-l1-course/tree/main/python/stepik_solutions).
