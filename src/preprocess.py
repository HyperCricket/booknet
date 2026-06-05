"""
File for preprocessing dataset into clean, ready to train data.
Steps:
    1. Load data into Pandas Dataframe
    2. Filter sparse users/books
    3. Build user-item matrix
    4. Convert to sparse matrix
"""

import pandas as pd
from scipy.sparse import csr_matrix

RATINGS_PATH = "../data/raw/goodbooks/ratings.csv"
MIN_AMT_USER_REVIEWS = 5
MIN_AMT_BOOK_REVIEWS = 10


def load(path: str = RATINGS_PATH) -> pd.DataFrame:
    """Function to load ratings csv to a dataframe

    :param path: path to csv file
    :type path: string
    :return: Dataframe of the csv file
    :rtype: pd.DataFrame

    """
    ratings = pd.read_csv(RATINGS_PATH)
    return ratings


def clean(ratings: pd.DataFrame) -> pd.DataFrame:
    """Filter out any users that have given less than 5 reviews and books that have less than 10 reviews

    :param ratings: Dataframe containing the ratings
    :type ratings: pd.DataFrame
    :return: Dataframe of the filtered ratings
    :rtype: pd.DataFrame
    """
    user_counts = ratings["user_id"].value_counts()
    book_counts = ratings["book_id"].value_counts()

    filtered_ratings = ratings[
        (ratings["user_id"].map(user_counts) >= MIN_AMT_USER_REVIEWS)
        & (ratings["book_id"].map(book_counts) >= MIN_AMT_BOOK_REVIEWS)
    ]

    return filtered_ratings


def build_matrix(ratings: pd.DataFrame) -> pd.DataFrame:
    """Create the pivot table where: rows are the users, cols are the books, and the indices are the ratings

    :param ratings: Dataframe containing the ratings
    :type ratings: pd.DataFrame
    :return: Dataframe of the pivot table matrix
    :rtype: pd.DataFrame
    """
    matrix = ratings.pivot_table(index="user_id", columns="book_id", values="rating")

    return matrix


def to_sparse(matrix: pd.DataFrame) -> csr_matrix:
    """Fill NaNs with 0 and convert to scipy sparse matrix to reduce memory consumption

    :param matrix: pivot table
    :type matrix: pd.DataFrame
    :return: csr_matrix for better memory usage
    :rtype: csr_matrix
    """
    sparse = csr_matrix(matrix.fillna(0))
    return sparse


def preprocess(path: str = RATINGS_PATH) -> csr_matrix:
    """Run full preprocessing pipeline in this method

    :param path: path to csv file
    :type path: string
    :return: csr_matrix for better memory usage
    :rtype: csr_matrix
    """
    ratings = load(RATINGS_PATH)
    filtered_ratings = clean(ratings)
    table = build_matrix(filtered_ratings)
    sparse = to_sparse(table)
    print("Successfully converted ratings.csv into a sparse matrix!")
    return sparse


if __name__ == "__main__":
    preprocess()
