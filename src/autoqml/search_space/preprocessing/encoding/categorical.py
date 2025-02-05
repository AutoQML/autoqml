from typing import Optional, Union

from optuna import Trial
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer

from autoqml.constants import InputData, TargetData
from autoqml.search_space import SearchSpace, Configuration
from autoqml.search_space.base import TunableMixin, IdentityTransformer
from autoqml.search_space.preprocessing.encoding.util import split_types


class CategoricalEncoder(BaseEstimator, TransformerMixin, TunableMixin):
    def __init__(
        self,
        categories: Union[str, list[str]] = 'auto',
        min_frequency: Optional[Union[int, float]] = None,
        max_categories: Optional[int] = None
    ):
        """
        categories ‘auto’ or a list of array-like, default=’auto’
        Categories (unique values) per feature:
        ‘auto’ : Determine categories automatically from the training data.
        list : categories[i] holds the categories expected in the ith column.
        The passed categories should not mix strings and numeric values within
        a single feature, and should be sorted in case of numeric values.
        The used categories can be found in the categories_ attribute.

        min_frequency int or float, default=None
        Specifies the minimum frequency below which a category will be
        considered infrequent.
        If int, categories with a smaller cardinality will be considered
        infrequent.
        If float, categories with a smaller cardinality than
        min_frequency * n_samples will be considered infrequent.

        max_categoriesint, default=None
        Specifies an upper limit to the number of output categories for each
        input feature when considering infrequent categories. If there are
        infrequent categories, max_categories includes the category
        representing the infrequent categories along with the frequent
        categories. If None, there is no limit to the number of output
        features.

        max_categories do not take into account missing or unknown categories.
        Setting unknown_value or encoded_missing_value to an integer will
        increase the number of unique integer codes by one each. This can
        result in up to max_categories + 2 integer codes.
        """
        self.categories = categories
        self.min_frequency = min_frequency
        self.max_categories = max_categories

    def fit(self, X: InputData, y: TargetData):
        from sklearn.preprocessing import OrdinalEncoder

        num_cols, cat_cols = split_types(X)
        self.estimator = ColumnTransformer(
            transformers=[
                ("num", IdentityTransformer(), num_cols),
                (
                    "cat",
                    OrdinalEncoder(
                        categories=self.categories,
                        # TODO: After sklearn version 1.3.0 switch on
                        # min_frequency and max_categories.
                        # min_frequency=self.min_frequency,
                        # max_categories=self.max_categories,
                    ),
                    cat_cols
                ),
            ]
        )

        self.estimator.fit(X, y)
        return self

    def transform(self, X: InputData) -> InputData:
        if not hasattr(self, 'estimator'):
            raise NotImplementedError()

        X = self.estimator.transform(X)
        return X

    def sample_configuration(
        self, trial: Trial, defaults: Configuration
    ) -> Configuration:
        return {}
