"""
HyperPhoenixCV - Возобновляемый поиск гиперпараметров с поддержкой чекпоинтов.

Этот модуль предоставляет класс HyperPhoenixCV, который расширяет функциональность
GridSearchCV из scikit-learn, добавляя поддержку чекпоинтов, случайного поиска
и байесовской оптимизации для ускорения поиска оптимальных гиперпараметров.
"""

import os
import random
import numpy as np
import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import ParameterGrid, cross_validate, StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

class HyperPhoenixCV(BaseEstimator):
    """
    Возобновляемый поиск гиперпараметров с поддержкой чекпоинтов и байесовской оптимизации.
    Поддерживает полный перебор, случайный поиск и байесовскую оптимизацию.

    Пример использования:
    # Создание объекта
    hp = HyperPhoenixCV(
        estimator=combat_pipeline,
        param_grid={
            'tfidf__max_features': [8000, 12000, 15000],
            'tfidf__ngram_range': [(1,1), (1,2)],
            'clf__C': [0.001, 0.01, 0.1],
            'clf__penalty': ['l1','l2'],
            'clf__solver': ['liblinear', 'saga'],
            'clf__class_weight': [None, 'balanced']
        },
        scoring=['f1', 'accuracy'],
        cv=5,
        n_jobs=-2,
        checkpoint_path="experiment_checkpoint.pkl",
        results_csv="experiment_results.csv",
        verbose=True
    )

    # Запуск поиска
    hp.fit(X, y)

    # Если процесс был прерван, запустите снова с тем же checkpoint_path:
    hp.fit(X, y)  # Продолжит с последней сохраненной точки!

    # Получение результатов
    print("Лучшие параметры:", hp.best_params_)
    print("Лучший скор:", hp.best_score_)

    # Топ-10 результатов
    top_10 = hp.get_top_results(10)
    print(top_10)

    # Удалить чекпоинт вручную
    hp.clear_checkpoint()
    """

    def __init__(
        self,
        estimator,
        param_grid: dict,
        scoring: str | list[str] = 'f1',
        cv: int = 5,
        n_jobs: int = 1,
        checkpoint_path: str = "hyperphoenix_checkpoint.pkl",
        results_csv: str = "hyperphoenix_results.csv",
        verbose: bool = True,
        clear_checkpoint: bool = False,
        random_search: bool = False,
        n_iter: int = 10,
        random_state: int | None = None,
        use_bayesian_optimization: bool = False,
        bayesian_optimizer = None,
        refit: bool = True,
    ):
        """
        Инициализирует HyperPhoenixCV.

        Parameters:
        -----------
        estimator : sklearn estimator
            Модель/пайплайн для подбора гиперпараметров
        param_grid : dict
            Словарь параметров для перебора
        scoring : str or list of str
            Метрики для оценки (например, 'f1', 'accuracy' или ['f1', 'accuracy'])
        cv : int
            Количество фолдов для кросс-валидации
        n_jobs : int
            Количество процессов для параллельных вычислений
        checkpoint_path : str
            Путь к файлу чекпоинта
        results_csv : str
            Путь к CSV файлу с результатами
        verbose : bool
            Печатать ли прогресс
        clear_checkpoint : bool
            Удалять ли существующий чекпоинт при инициализации
        random_search : bool
            Использовать ли случайный перебор вместо полного
        n_iter : int
            Количество случайных комбинаций (если random_search=True)
        random_state : int, optional
            Фиксация случайных чисел для воспроизводимости
        use_bayesian_optimization : bool
            Использовать ли байесовскую оптимизацию (предсказательный отбор параметров)
        bayesian_optimizer : sklearn regressor, optional
            Модель, которая предсказывает, какие параметры будут лучше
            (по умолчанию RandomForestRegressor)
        refit : bool, default=True
            Обучать ли лучшую модель на всем датасете после поиска.
            Если True, после завершения поиска гиперпараметров будет вызван
            `best_estimator_.fit(X, y)`.
        """
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring if isinstance(scoring, list) else [scoring]
        self.cv = cv
        self.n_jobs = n_jobs
        self.checkpoint_path = checkpoint_path
        self.results_csv = results_csv
        self.verbose = verbose
        self.random_search = random_search
        self.n_iter = n_iter
        self.random_state = random_state
        self.use_bayesian_optimization = use_bayesian_optimization
        self.bayesian_optimizer = (
            bayesian_optimizer or RandomForestRegressor(n_estimators=20, random_state=42)
        )
        self.label_encoders = {}
        self.refit = refit

        # Удаляем чекпоинт, если указано
        if clear_checkpoint and os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            if self.verbose:
                print(f"Удалён чекпоинт: {checkpoint_path}")

    def _generate_param_list(self) -> list[dict]:
        """
        Генерирует список параметров: полный перебор или случайный.

        Returns:
        --------
        list[dict]: Список комбинаций параметров для тестирования.
        """
        all_params = list(ParameterGrid(self.param_grid))

        if self.random_search:
            if self.random_state is not None:
                random.seed(self.random_state)

            if len(all_params) <= self.n_iter:
                if self.verbose:
                    print(
                        f"Всего комбинаций ({len(all_params)}) <= n_iter ({self.n_iter}). "
                        f"Используем все."
                    )
                return all_params
            # else is unnecessary after return
            selected_params = random.sample(all_params, self.n_iter)
            if self.verbose:
                print(
                    f"Выбрано {len(selected_params)} случайных комбинаций "
                    f"из {len(all_params)} возможных."
                )
            return selected_params
        else:
            if self.verbose:
                print(f"Полный перебор: {len(all_params)} комбинаций.")
            return all_params

    def _load_checkpoint(self) -> list[dict]:
        """
        Загружает результаты из чекпоинта.

        Returns:
        --------
        list[dict]: Список результатов из чекпоинта.
        """
        if os.path.exists(self.checkpoint_path):
            results = joblib.load(self.checkpoint_path)
            if self.verbose:
                print(f"Загружено {len(results)} завершённых комбинаций из чекпоинта.")
                # Выводим текущие лучшие результаты из чекпоинта
                if results:
                    valid_results = [r for r in results if 'error' not in r]
                    if valid_results:
                        # Сортируем по первой метрике
                        best_result = max(valid_results,
                                          key=lambda x: x.get(f'mean_test_{self.scoring[0]}',
                                                              float('-inf')))
                        print(f"Текущий лучший результат из чекпоинта:")
                        score_key = f'mean_test_{self.scoring[0]}'
                        std_key = f'std_test_{self.scoring[0]}'
                        print(f"   score: {best_result.get(score_key, 0):.4f} ± "
                              f"{best_result.get(std_key, 0):.4f}")
                        if len(self.scoring) > 1:
                            for metric in self.scoring[1:]:
                                mean_key = f'mean_test_{metric}'
                                std_key = f'std_test_{metric}'
                                if mean_key in best_result and std_key in best_result:
                                    print(f"   {metric}: {best_result[mean_key]:.4f} ± "
                                          f"{best_result[std_key]:.4f}")
                        print(f"   Параметры: {best_result.get('params', {})}")
            return results
        return []

    def _save_checkpoint(self, results: list[dict]):
        """
        Сохраняет результаты в чекпоинт.

        Parameters:
        -----------
        results : list[dict]
            Список результатов для сохранения.
        """
        joblib.dump(results, self.checkpoint_path)

    def _format_scores(self, cv_results: dict[str, np.ndarray]) -> dict[str, any]:
        """
        Форматирует результаты кросс-валидации.

        Parameters:
        -----------
        cv_results : dict[str, np.ndarray]
            Результаты кросс-валидации от cross_validate.

        Returns:
        --------
        dict[str, any]: Отформатированные результаты.
        """
        scores = {}
        for metric in self.scoring:
            test_metric = f'test_{metric}'
            if test_metric in cv_results:
                scores[f'mean_test_{metric}'] = float(cv_results[test_metric].mean())
                scores[f'std_test_{metric}'] = float(cv_results[test_metric].std())
                scores[f'scores_{metric}'] = cv_results[test_metric].tolist()
        return scores

    def _encode_params(self, params_list: list[dict]) -> np.ndarray:
        """
        Кодирует список параметров в числовую матрицу.

        Parameters:
        -----------
        params_list : list[dict]
            Список параметров для кодирования.

        Returns:
        --------
        np.ndarray: Закодированные параметры в виде матрицы.
        """
        if not params_list:
            return np.array([]).reshape(0, -1)

        df = pd.DataFrame(params_list)
        X = df.copy()

        for col in X.columns:
            if X[col].dtype == 'object':
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
                else:
                    X[col] = self.label_encoders[col].transform(X[col].astype(str))
            else:
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

        return X.values

    def _suggest_next_params(
        self,
        all_param_combinations: list[dict],
        completed_results: list[dict],
    ) -> list[dict]:
        """
        Сортирует оставшиеся параметры по предсказанной метрике (если используется байесовская оптимизация).

        Parameters:
        -----------
        all_param_combinations : list[dict]
            Все возможные комбинации параметров.
        completed_results : list[dict]
            Уже завершённые результаты.

        Returns:
        --------
        list[dict]: Отсортированный список параметров.
        """
        if not self.use_bayesian_optimization or not completed_results:
            return all_param_combinations

        # Обучаем модель на завершённых результатах
        completed_params = [r['params'] for r in completed_results]
        completed_scores = [r[f'mean_test_{self.scoring[0]}'] for r in completed_results]

        X_train = self._encode_params(completed_params)
        y_train = np.array(completed_scores)

        if X_train.size == 0 or len(y_train) == 0:
            return all_param_combinations

        self.bayesian_optimizer.fit(X_train, y_train)

        # Предсказываем для оставшихся
        X_remaining = self._encode_params(all_param_combinations)
        if X_remaining.size == 0:
            return all_param_combinations

        predicted_scores = self.bayesian_optimizer.predict(X_remaining)

        # Сортируем по убыванию предсказанного результата
        sorted_indices = np.argsort(predicted_scores)[::-1]
        return [all_param_combinations[i] for i in sorted_indices]

    def fit(self, X, y, groups=None):
        """
        Выполняет подбор гиперпараметров с сохранением промежуточных результатов.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Обучающие данные.
        y : array-like of shape (n_samples,)
            Целевые значения.
        groups : array-like of shape (n_samples,), default=None
            Группы для групповой кросс-валидации (если используется).

        Returns:
        --------
        self : object
            Возвращает экземпляр класса.
        """
        # Загружаем прогресс
        all_results = self._load_checkpoint()

        # Генерируем все комбинации параметров (полный или случайный перебор)
        param_list = self._generate_param_list()
        if self.verbose:
            print(f"Всего комбинаций: {len(param_list)}")

        # Исключаем уже обработанные
        completed_params = [r['params'] for r in all_results if 'params' in r]
        remaining_params = [p for p in param_list if p not in completed_params]
        if self.verbose:
            print(f"Осталось обработать: {len(remaining_params)}")

        # Если используется байесовская оптимизация — сортируем оставшиеся параметры по предсказанию
        if self.use_bayesian_optimization:
            remaining_params = self._suggest_next_params(remaining_params, all_results)
            if self.verbose:
                print("Оставшиеся параметры отсортированы по предсказанной метрике.")

        # --- Определяем CV ---

        if isinstance(self.cv, int):
            classification_metrics = {
                'accuracy', 'balanced_accuracy', 'f1', 'precision', 'recall',
                'f1_macro', 'f1_micro', 'f1_weighted', 'precision_macro',
                'precision_micro', 'precision_weighted', 'recall_macro',
                'recall_micro', 'recall_weighted', 'jaccard', 'roc_auc'
            }

            if any(m in classification_metrics for m in self.scoring):
                cv_splitter = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42)
            else:
                cv_splitter = KFold(n_splits=self.cv, shuffle=True, random_state=42)
        else:
            cv_splitter = self.cv
        # --- CV определён ---

        # Перебираем оставшиеся параметры
        for i, params in enumerate(remaining_params, start=1):
            if self.verbose:
                print(f"\n[{i}/{len(remaining_params)}] Тестируем: {params}")

            try:
                estimator_with_params = self.estimator.set_params(**params)

                scores = cross_validate(
                    estimator_with_params, X, y,
                    cv=cv_splitter,
                    scoring=self.scoring,
                    n_jobs=self.n_jobs
                )

                result = {
                    'params': params,
                    **self._format_scores(scores)
                }
                all_results.append(result)

                # Обновляем модель байесовской оптимизации, если используется
                if self.use_bayesian_optimization:
                    # Необязательно обновлять каждый раз — можно раз в N итераций
                    pass  # Модель обновляется при следующем вызове _suggest_next_params

                self._save_checkpoint(all_results)

                if self.verbose:
                    current_metrics = []
                    for metric in self.scoring:
                        mean_key = f'mean_test_{metric}'
                        std_key = f'std_test_{metric}'
                        if mean_key in result and std_key in result:
                            current_metrics.append(
                                f"{metric}: {result[mean_key]:.4f} ± {result[std_key]:.4f}"
                            )
                    current_str = " | ".join(current_metrics)

                    metric_key = f'mean_test_{self.scoring[0]}'
                    if metric_key in result:
                        best_score = max(
                            r[metric_key] for r in all_results if metric_key in r
                        )
                        best_metrics = []
                        for metric in self.scoring:
                            metric_key_other = f'mean_test_{metric}'
                            if metric_key_other in result:
                                best_other = max(
                                    r[metric_key_other]
                                    for r in all_results
                                    if metric_key_other in r
                                )
                                best_metrics.append(f"{metric}: {best_other:.4f}")
                        best_str = " | ".join(best_metrics)
                        print(f"Сохранено. Текущие: {current_str} | Лучшие: {best_str}")

            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Ошибка: {e}")
                all_results.append({
                    'params': params,
                    'error': str(e)
                })
                self._save_checkpoint(all_results)

        # Сохраняем результаты в CSV
        self._save_results_to_csv(all_results)

        # Сохраняем атрибуты для совместимости с GridSearchCV
        self.cv_results_ = self._format_cv_results(all_results)
        self.best_params_ = self._get_best_params(all_results)
        self.best_score_ = self._get_best_score(all_results)

        self.best_estimator_ = self.estimator.set_params(**self.best_params_)
        if self.refit:
            self.best_estimator_.fit(X, y)

        if self.verbose:
            print(f"\nВсе результаты сохранены в {self.results_csv}")
            print(f"Лучший результат ({self.scoring[0]}): {self.best_score_:.4f}")
            if self.random_search:
                total_grid = len(list(ParameterGrid(self.param_grid)))
                print(
                    f"Использован случайный перебор: {self.n_iter} из {total_grid} "
                    f"возможных комбинаций ({self.n_iter/total_grid*100:.2f}%)"
                )

        return self

    def predict(self, X):
        """
        Предсказания с помощью лучшей модели.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Данные для предсказания.

        Returns:
        --------
        y_pred : array-like of shape (n_samples,)
            Предсказанные значения.
        """
        check_is_fitted(self, 'best_estimator_')
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        """
        Вероятности классов (если лучшая модель поддерживает predict_proba).

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Данные для предсказания.

        Returns:
        --------
        y_proba : array-like of shape (n_samples, n_classes)
            Вероятности классов.
        """
        check_is_fitted(self, 'best_estimator_')
        return self.best_estimator_.predict_proba(X)

    def score(self, X, y):
        """
        Оценка лучшей модели на данных X, y.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Данные для оценки.
        y : array-like of shape (n_samples,)
            Истинные значения.

        Returns:
        --------
        score : float
            Значение метрики (по умолчанию используется метрика scoring[0]).
        """
        check_is_fitted(self, 'best_estimator_')
        return self.best_estimator_.score(X, y)

    def _format_cv_results(self, results: list[dict]) -> dict[str, np.ndarray]:
        """
        Форматирует результаты в формат, совместимый с GridSearchCV.

        Parameters:
        -----------
        results : list[dict]
            Список результатов.

        Returns:
        --------
        dict[str, np.ndarray]: Форматированные результаты.
        """
        valid_results = [r for r in results if 'error' not in r]
        if not valid_results:
            return {}

        # Создаём словарь с результатами
        cv_results = {'params': [r['params'] for r in valid_results]}

        for metric in self.scoring:
            mean_key = f'mean_test_{metric}'
            std_key = f'std_test_{metric}'

            cv_results[mean_key] = np.array([r[mean_key] for r in valid_results])
            cv_results[std_key] = np.array([r[std_key] for r in valid_results])

        return cv_results

    def _get_best_params(self, results: list[dict]) -> dict:
        """
        Получает лучшие параметры.

        Parameters:
        -----------
        results : list[dict]
            Список результатов.

        Returns:
        --------
        dict: Лучшие параметры.
        """
        valid_results = [r for r in results if 'error' not in r]
        if not valid_results:
            return {}

        # Сортируем по первой метрике
        best_result = max(valid_results,
                         key=lambda x: x[f'mean_test_{self.scoring[0]}'])
        return best_result['params']

    def _get_best_score(self, results: list[dict]) -> float:
        """
        Получает лучший скор.

        Parameters:
        -----------
        results : list[dict]
            Список результатов.

        Returns:
        --------
        float: Лучший скор.
        """
        valid_results = [r for r in results if 'error' not in r]
        if not valid_results:
            return 0.0

        best_result = max(valid_results,
                         key=lambda x: x[f'mean_test_{self.scoring[0]}'])
        return best_result[f'mean_test_{self.scoring[0]}']

    def _save_results_to_csv(self, results: list[dict]):
        """
        Сохраняет результаты в CSV.

        Parameters:
        -----------
        results : list[dict]
            Список результатов для сохранения.
        """
        valid_results = [r for r in results if 'error' not in r]
        if not valid_results:
            df = pd.DataFrame(columns=['params'])
            df.to_csv(self.results_csv, index=False)
            return

        # Формируем DataFrame
        rows = []
        for r in valid_results:
            row = {}
            # Добавляем параметры как отдельные колонки
            row.update(r['params'])
            # Добавляем метрики
            for metric in self.scoring:
                mean_key = f'mean_test_{metric}'
                std_key = f'std_test_{metric}'
                if mean_key in r:
                    row[mean_key] = r[mean_key]
                if std_key in r:
                    row[std_key] = r[std_key]
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(self.results_csv, index=False)

    def get_top_results(self, n: int = 10) -> pd.DataFrame:
        """
        Возвращает топ-N результатов.

        Parameters:
        -----------
        n : int
            Количество топ результатов для возврата.

        Returns:
        --------
        pd.DataFrame: Топ-N результатов.
        """
        if not hasattr(self, 'cv_results_') or not self.cv_results_:
            return pd.DataFrame()

        # Создаём DataFrame из результатов
        results = []
        for i in range(len(self.cv_results_['params'])):
            row = {}
            row.update(self.cv_results_['params'][i])
            for metric in self.scoring:
                row[f'mean_test_{metric}'] = self.cv_results_[f'mean_test_{metric}'][i]
                row[f'std_test_{metric}'] = self.cv_results_[f'std_test_{metric}'][i]
            results.append(row)

        df = pd.DataFrame(results)
        # Сортируем по первой метрике
        df = df.sort_values(f'mean_test_{self.scoring[0]}', ascending=False)
        return df.head(n)

    def clear_checkpoint(self):
        """
        Удаляет файл чекпоинта.
        """
        if os.path.exists(self.checkpoint_path):
            os.remove(self.checkpoint_path)
            if self.verbose:
                print(f"Удалён чекпоинт: {self.checkpoint_path}")

    def load_results_from_checkpoint(self, n: int = 10) -> pd.DataFrame:
        """
        Загружает результаты из чекпоинта и возвращает топ-N.
        Полезно, если fit() был прерван и CSV не был создан.

        Parameters:
        -----------
        n : int
            Количество топ результатов для возврата

        Returns:
        --------
        pd.DataFrame
            Топ-N результатов из чекпоинта
        """
        if not os.path.exists(self.checkpoint_path):
            if self.verbose:
                print(f"⚠️ Чекпоинт {self.checkpoint_path} не найден.")
            return pd.DataFrame()

        # Загружаем результаты из чекпоинта
        all_results = self._load_checkpoint()
        valid_results = [r for r in all_results if 'error' not in r]

        if not valid_results:
            if self.verbose:
                print("⚠️ В чекпоинте нет валидных результатов.")
            return pd.DataFrame()

        # Формируем DataFrame
        rows = []
        for r in valid_results:
            row = {}
            # Добавляем параметры как отдельные колонки
            row.update(r['params'])
            # Добавляем метрики
            for metric in self.scoring:
                mean_key = f'mean_test_{metric}'
                std_key = f'std_test_{metric}'
                if mean_key in r:
                    row[mean_key] = r[mean_key]
                if std_key in r:
                    row[std_key] = r[std_key]
            rows.append(row)

        df = pd.DataFrame(rows)

        # Сортируем по первой метрике (та, по которой ищем лучшее)
        df = df.sort_values(f'mean_test_{self.scoring[0]}', ascending=False)

        if self.verbose:
            print(f"Загружено {len(df)} валидных результатов из чекпоинта.")
            print(f"Лучший {self.scoring[0]}: {df.iloc[0][f'mean_test_{self.scoring[0]}']:.4f}")

        return df.head(n)
