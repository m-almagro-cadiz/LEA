import numpy as np
from itertools import combinations
from itertools import groupby
from bisect import bisect_left
from importlib import import_module
import nltk
import pylcs
import jaro
import swalign

def tokenize_function(features: dict, operations: list, p: float, columns: list, lea_embedding_n: int=50, get_positions: bool=False) -> dict:
    """
    Transform two lists of texts into a list of pairs
    
    Args:
       features (dict): dictionary with all the features
       lea_embedding_n (int): number of discrete similarity values to be assigned
       get_positions (bool): flag indicating if positions within a learnable matrix instead similarity values are assigned
    Return:
       dict
    """
    features = augment(operations, p, columns, features)
    features = tokenizer(concatenate_inputs(features[columns[0]], features[columns[1]]), padding="max_length", truncation=True)
    features["label"] = [np.array([d]) for d in features["label"]]
    word_ids = [features[i].word_ids for i in range(len(features["input_ids"]))]
    features = add_lex_sim_to_batch(tokenizer, features, word_ids, metric="lev", lea_embedding_n=lea_embedding_n, get_positions=get_positions)
    return features

def augment(operations: list, p: float, columns: list, batch: dict):
    """Build the sequential list of raw data augmentation operations"""
    augs = []
    for aug_op in operations:
        mod = import_module("nlpaug.augmenter.char")
        augmenter = getattr(mod, aug_op["type"])
        aug = augmenter(**aug_op.get("args", {}))
        augs.append(aug)
    if np.random.choice([False, True], p=[1 - p, p]):
        for col in columns:
            batch[col] = random.choice(augs).augment(batch[col])    
    return batch

def concatenate_inputs(texts1: list, texts2: list) -> list:
    """
    Transform two lists of texts into a list of pairs
    
    Args:
       texts1 (list): list of texts
       texts2 (list): list of texts
    Return:
       list
    """
    return [((text1 if text1 else ""), (text2 if text2 else "")) for text1,text2 in zip(texts1, texts2)]

def add_lex_sim_to_batch(tokenizer, features: dict, word_ids: list, metric: str="", lea_embedding_n: int=50, get_positions: bool=False, self_attention: bool=False, exclude_exacts: bool=False) -> dict:
    """
    Add the lexical similarity values per token to the tokenized features

    Args:
       features (dict): dictionary with all the features
       word_ids (list): list of word ids to estimate the similarity value between whole words
       metric (str): metric used to estimate the similarity value
       lea_embedding_n (int): number of discrete similarity values to be assigned
       get_positions (bool): flag indicating if positions within a learnable matrix instead similarity values are assigned
       self_attention (bool): flag indicating if similarities values are also assigned to words within the same sentence
       exclude_exacts (bool): flag indicting if exact matches are discarded and a null similarity values is assigned
    Return:
       dict
    """
    valid_values = None
    # Discretize the range [0,1] into N values
    if lea_embedding_n > 0:
        valid_values = np.round(np.linspace(0, 1, lea_embedding_n), len(str(lea_embedding_n)))
    # Compute similarity values for each instance
    if type(features["input_ids"][0]) is int:
        sim_matrix = compute_instance_similarity_matrix(valid_values, tokenizer, features["input_ids"], word_ids, features["token_type_ids"], metric, get_positions, self_attention, exclude_exacts)
        features["lex_sim"] = sim_matrix
    else:
        features["lex_sim"] = list()
        for p in range(len(features["input_ids"])):
            sim_matrix = compute_instance_similarity_matrix(valid_values, tokenizer, features["input_ids"][p], word_ids[p], features["token_type_ids"][p], metric, get_positions, self_attention, exclude_exacts)
            features["lex_sim"].append(sim_matrix)
    return features

def compute_instance_similarity_matrix(valid_values: list, tokenizer, input_ids: list, word_ids: list, token_type_ids: list, metric: str, get_positions: bool=False, self_attention: bool=False, exclude_exacts: bool=False) -> list:
    """
    Compute the similarity matrix of all tokens against all for each instance.
    Args:
       valid_values (list): allowed values. If provided, values are rounded to the closest allowed value; if None, all values are used
       tokenizer (LocalFileSystemDataModule)
       input_ids (list): list of input ids
       word_ids (list): list of word ids
       token_type_ids (list): list of token type ids
       metric (str): metric used to estimate the similarity value
       get_positions (bool): flag indicating if positions within a learnable matrix instead similarity values are assigned
       self_attention (bool): flag indicating if similarities values are also assigned to words within the same sentence
       exclude_exacts (bool): flag indicting if exact matches are discarded and a null similarity values is assigned
    Returns:
       list: list of lists with similarity values per token
    """
    # Get all words per token type id
    word_dict = get_words_from_tokenids(tokenizer, input_ids, word_ids, token_type_ids)
    # Estimate all similarity values for each word pair
    sim_dict = get_sim_from_word_dict(word_dict, metric, valid_values, get_positions)
    # Fill the token-level similarity matrix
    sim_matrix = get_token_level_sim_matrix(word_ids, token_type_ids, word_dict, sim_dict, self_attention, exclude_exacts)
    # Add the pre-computed similarity matrix to the features
    return sim_matrix

def get_sim_from_word_dict(word_dict: dict, metric: str, valid_values: list = None, get_positions: bool=False) -> dict:
    """Generate word pair combinations and compute similarity values between all word pairs
    Args:
       word_dict (dict): dictionary contaning words per token type id
       valid_values (list): list of discrete values to approximate the similarity values
       get_positions (bool): flag indicating if positions within a learnable matrix instead similarity values are assigned
    Returns:
        dict: dictionary containing a similarity value per word pair
    """
    sim_dict = dict()
    # Get the set of words including all token types
    word_set = list(set([word for word_list in word_dict.values() for word in word_list]))
    # Produce all possible word pairs
    word_pairs = list(combinations(word_set, 2)) + [(word, word) for word in word_set]
    # Compute text similarity for each pair
    for word1, word2 in word_pairs:
        if (metric == "jaccard"):
            sim_val = compute_jaccard_similarity(word1, word2)
        elif (metric == "lcs"):
            sim_val = compute_lcs_similarity(word1, word2)
        elif (metric == "smith"):
            sim_val = compute_smith_similarity(word1, word2)
        elif (metric == "jaro"):
            sim_val = compute_jaro_similarity(word1, word2)
        else:
            sim_val = compute_levenshtein_similarity(word1, word2)
        # Approximate the value
        if valid_values is not None:
            sim_val = take_closest(valid_values, sim_val, get_positions)
        sim_dict[(word1, word2)] = sim_val
        sim_dict[(word2, word1)] = sim_val
    return sim_dict


def get_token_level_sim_matrix(wordids_list: list, typeids_list: list, word_dict: dict, sim_dict: dict, self_attention: bool=False, exclude_exacts: bool=False) -> list:
    """Fill the token-level similarity matrix considering word ids and word pair similarity values
    Args:
       wordids_list (list): list of word ids
       typeids_list (list): list of token type ids
       word_dict (dict): dictionary contaning words per token type id
       sim_dict (dict): dictionary containing similarity values for word pairs
       self_attention (bool): flag indicating if similarities values are also assigned to words within the same sentence
       exclude_exacts (bool): flag indicting if exact matches are discarded and a null similarity values is assigned
    Returns:
        list: list of lists with similarity values per token
    """
    # Initialize token-level similarity matrix
    sim_matrix = list()
    # Fill the token-level similarity matrix
    for word_id1, token_type_id1 in zip(wordids_list, typeids_list):
        # Initialize a row of the token-level similarity matrix
        sim_matrix_row = list()
        # Get all similarity values for one token
        for word_id2, token_type_id2 in zip(wordids_list, typeids_list):
            # Set zero values for special tokens (word id below 0) and all tokens pertaining to the same token type id
            if (word_id1 is not None and word_id2 is not None and word_id1 >= 0 and word_id2 >= 0 and (self_attention or token_type_id1 != token_type_id2) and (not exclude_exacts or word_id1 != word_id2)):
                sim_matrix_row.append(sim_dict[(word_dict[token_type_id1][word_id1], word_dict[token_type_id2][word_id2])])
            else:
                sim_matrix_row.append(0)
        sim_matrix.append(sim_matrix_row)
    return sim_matrix

def take_closest(values: list, n: int, get_positions: bool=False) -> list:
    """
    Assumes myList is sorted. Returns closest value to myNumber.
    If two numbers are equally close, return the smallest number.

    Args:
       values (list): list of floats
       n (int): number of discrete values
       get_positions (bool):  return the positions of the discrete values instead of the values
    """
    pos = bisect_left(values, n)
    if pos == 0:
        return values[0]
    if pos == len(values):
        return values[-1]
    before = values[pos - 1]
    after = values[pos]
    if after - n < n - before:
        return pos if get_positions else after
    else:
        return pos - 1 if get_positions else before

def get_words_from_tokenids(tokenizer, tokenids_list: list, wordids_list: list, typeids_list: list) -> dict:
    """Decode token ids and get separate words
    Args:
        tokenizer: pre-trained tokenizer
        tokenids_list (list): list of token ids
        wordids_list (list): list of word ids
        typeids_list (list): list of token type ids
    Returns:
        dict: words associated with word ids per token type id
    """
    # Dictionary containing type ids associated with ids and words
    word_dict = {type: [] for type in set(typeids_list)}
    # Group consecutive ids and generate tuples with ids and number of repetitions
    grouped_wordids = [(wid, sum(1 for i in g)) for wid, g in groupby(wordids_list)]
    # Get consecutive ids
    wordids = [wid for wid, s in grouped_wordids]
    # Compute indices for ids
    position_wordids = np.cumsum([s for wid, s in grouped_wordids])
    for i in range(len(wordids)):
        if wordids[i] is not None and wordids[i] >= 0:
            tokenids = tokenids_list[position_wordids[i - 1] if i > 0 else 0 : position_wordids[i]]
            word_dict[typeids_list[position_wordids[i]]].append(tokenizer.decode(tokenids))
    return word_dict

def compute_levenshtein_similarity(word1: str, word2: str) -> float:
    """Compute the levenshtein similarity between the word 1 and word 2, which is based on the normalized levenshtein distance.
    Args:
       word1 (str): First word
       word2 (str): Second word

    Return:
       float: levenshtein similarity value
    """
    return 1 - nltk.edit_distance(word1, word2) / max(len(word1), len(word2))


def compute_jaccard_similarity(word1: str, word2: str) -> float:
    """Compute the Jaccard similarity between the word 1 and word 2.
    Args:
       word1 (str): First word
       word2 (str): Second word

    Return:
       float: Jaccard similarity value
    """
    intersection = len(list(set(word1).intersection(word2)))
    union = (len(set(word1)) + len(set(word2))) - intersection
    return float(intersection) / union


def compute_lcs_similarity(word1: str, word2: str) -> float:
    """Compute the similarity based on the Least Common Subsequence between the word 1 and word 2.
    Args:
       word1 (str): First word
       word2 (str): Second word

    Return:
       float: LCS similarity value
    """
    return pylcs.lcs_sequence_length(word1, word2) / max(len(word1), len(word2))


def compute_jaro_similarity(word1: str, word2: str) -> float:
    """Compute the Jaro-Winkler similarity between the word 1 and word 2.
    Args:
       word1 (str): First word
       word2 (str): Second word

    Return:
       float: Jaro-Winkler similarity value
    """
    return jaro.jaro_winkler_metric(word1, word2)


def compute_smith_similarity(word1: str, word2: str) -> float:
    """Compute the Smith-Waterman similarity between the word 1 and word 2.
    Args:
       word1 (str): First word
       word2 (str): Second word

    Return:
       float: Smith-Waterman similarity value
    """
    match = 2
    mismatch = -1
    scoring = swalign.NucleotideScoringMatrix(match, mismatch)
    sw = swalign.LocalAlignment(scoring)
    alignment = sw.align(word1, word2)
    alignment.dump()
    return alignment.score
