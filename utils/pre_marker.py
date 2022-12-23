import pandas as pd
import numpy as np
from typing import Tuple, Callable
from ast import literal_eval

def entity_mask(df:pd.DataFrame, tokenizer) -> Tuple[np.ndarray, list, Callable]:
    wrapper_f = ''
    wrapper_b = ''
    switch = 'f'
    for w in tokenizer.unk_token:
        if not w.isalpha():
            if switch == 'f':
                wrapper_f += w
            elif switch == 'b':
                wrapper_b += w
        else:
            switch = 'b'
            
    special_token_list = []
    for token in ['OBJ-PER', 'OBJ-ORG', 'OBJ-DAT', 'OBJ-LOC', 'OBJ-POH', 'OBJ-NOH',
                'SUBJ-PER', 'SUBJ-ORG', 'SUBJ-LOC']:
        special_token_list.append(wrapper_f + token + wrapper_b)
    tokenizer.add_special_tokens({"additional_special_tokens":special_token_list})
    
    sentences_list = []
    entity_embedding_hint = []
    for _, row in df.iterrows():
        subject_dict = literal_eval(row['subject_entity'])
        object_dict = literal_eval(row['object_entity'])
        sentence = row['sentence']
        if subject_dict['start_idx'] <= object_dict['start_idx']: # 만약 subject가 앞에 있을 경우
            first_idx_s = subject_dict['start_idx']
            first_idx_e = subject_dict['end_idx']
            second_idx_s = object_dict['start_idx']
            second_idx_e = object_dict['end_idx']
            first_word =  'SUBJ-'+subject_dict['type']
            second_word =  'OBJ-'+object_dict['type']
        else:
            first_idx_s = object_dict['start_idx']
            first_idx_e = object_dict['end_idx']
            second_idx_s = subject_dict['start_idx']
            second_idx_e = subject_dict['end_idx']
            first_word = 'OBJ-'+object_dict['type']
            second_word = 'SUBJ-'+subject_dict['type']
            
        entity_embedding_hint.append((
            wrapper_f + first_word + wrapper_b,
            wrapper_f + second_word + wrapper_b
        ))
        
        sentence = sentence[:first_idx_s] + wrapper_f + first_word + wrapper_b + sentence[first_idx_e+1:]
        second_idx_s += len(wrapper_f) + len(first_word) + len(wrapper_b) - (first_idx_e + 1 - first_idx_s)
        second_idx_e += len(wrapper_f) + len(first_word) + len(wrapper_b) - (first_idx_e + 1 - first_idx_s)

        sentence = sentence[:second_idx_s] + wrapper_f + second_word + wrapper_b + sentence[second_idx_e+1:]
        sentences_list.append(sentence)

    return np.array(sentences_list), entity_embedding_hint, tokenizer

def entity_marker(df:pd.DataFrame, tokenizer) -> Tuple[np.ndarray, list, Callable]:
    wrapper_f = ''
    wrapper_b = ''
    switch = 'f'
    for w in tokenizer.unk_token:
        if not w.isalpha():
            if switch == 'f':
                wrapper_f += w
            elif switch == 'b':
                wrapper_b += w
        else:
            switch = 'b'

    special_token_list = []
    for token in ['E1', '/E1', 'E2', '/E2']:
        special_token_list.append(wrapper_f + token + wrapper_b)
    tokenizer.add_special_tokens({"additional_special_tokens":special_token_list})

    sentences_list = []
    entity_embedding_hint = []
    for _, row in df.iterrows():
        subject_dict = literal_eval(row['subject_entity'])
        object_dict = literal_eval(row['object_entity'])
        sentence = row['sentence']
        if subject_dict['start_idx'] <= object_dict['start_idx']: # 만약 subject가 앞에 있을 경우
            first_idx_s = subject_dict['start_idx']
            first_idx_e = subject_dict['end_idx']
            second_idx_s = object_dict['start_idx']
            second_idx_e = object_dict['end_idx']
            first_word = 'E1'
            second_word = 'E2'
        else:
            first_idx_s = object_dict['start_idx']
            first_idx_e = object_dict['end_idx']
            second_idx_s = subject_dict['start_idx']
            second_idx_e = subject_dict['end_idx']
            first_word = 'E2'
            second_word = 'E1'

        # 두 임베딩된 단어를 넣어주기
        entity_embedding_hint.append((
            wrapper_f + first_word + wrapper_b + sentence[first_idx_s:first_idx_e+1] + wrapper_f + '/' + first_word + wrapper_b,
            wrapper_f + second_word + wrapper_b + sentence[second_idx_s:second_idx_e+1] + wrapper_f + '/' + second_word + wrapper_b
        ))

        sentence = sentence[:first_idx_s] + wrapper_f + first_word + wrapper_b + sentence[first_idx_s:]
        first_idx_e += len(wrapper_f) + len(first_word) + len(wrapper_b)
        second_idx_s += len(wrapper_f) + len(first_word) + len(wrapper_b)
        second_idx_e += len(wrapper_f) + len(first_word) + len(wrapper_b)

        sentence = sentence[:first_idx_e+1] + wrapper_f + '/' + first_word + wrapper_b + sentence[first_idx_e+1:]
        second_idx_s += len(wrapper_f) + len('/') + len(first_word) + len(wrapper_b)
        second_idx_e += len(wrapper_f) + len('/') + len(first_word) + len(wrapper_b)

        sentence = sentence[:second_idx_s] + wrapper_f + second_word + wrapper_b + sentence[second_idx_s:]
        second_idx_e += len(wrapper_f) + len(second_word) + len(wrapper_b)

        sentence = sentence[:second_idx_e+1] + wrapper_f + '/' + second_word + wrapper_b + sentence[second_idx_e+1:]
        sentences_list.append(sentence)

    return np.array(sentences_list), entity_embedding_hint, tokenizer

def typed_entity_marker(df:pd.DataFrame, tokenizer) -> Tuple[np.ndarray, list, Callable]:
    wrapper_f = '' 
    wrapper_b = '' 
    switch = 'f'
    for w in tokenizer.unk_token:
        if not w.isalpha():
            if switch == 'f':
                wrapper_f += w
            elif switch == 'b':
                wrapper_b += w
        else:
            switch = 'b'

    special_token_list = []
    for token in ['O:PER', 'O:ORG', 'O:DAT', 'O:LOC', 'O:POH', 'O:NOH',
                '/O:PER', '/O:ORG', '/O:DAT', '/O:LOC', '/O:POH', '/O:NOH',
                'S:PER', 'S:ORG', '/S:PER', '/S:ORG', 'S:LOC', '/S:LOC'
                ]:
        special_token_list.append(wrapper_f + token + wrapper_b)
    tokenizer.add_special_tokens({"additional_special_tokens":special_token_list})

    sentences_list = []
    entity_embedding_hint = []
    for _, row in df.iterrows():
        subject_dict = literal_eval(row['subject_entity'])
        object_dict = literal_eval(row['object_entity'])
        sentence = row['sentence']
        if subject_dict['start_idx'] <= object_dict['start_idx']: # 만약 subject가 앞에 있을 경우
            first_idx_s = subject_dict['start_idx']
            first_idx_e = subject_dict['end_idx']
            second_idx_s = object_dict['start_idx']
            second_idx_e = object_dict['end_idx']
            first_word =  'S:'+subject_dict['type']
            second_word =  'O:'+object_dict['type']
        else:
            first_idx_s = object_dict['start_idx']
            first_idx_e = object_dict['end_idx']
            second_idx_s = subject_dict['start_idx']
            second_idx_e = subject_dict['end_idx']
            first_word = 'O:'+object_dict['type']
            second_word = 'S:'+subject_dict['type']
            
        entity_embedding_hint.append((
            wrapper_f + first_word + wrapper_b + sentence[first_idx_s:first_idx_e+1] + wrapper_f + '/' + first_word + wrapper_b,
            wrapper_f + second_word + wrapper_b + sentence[second_idx_s:second_idx_e+1] + wrapper_f + '/' + second_word + wrapper_b
        ))

        sentence = sentence[:first_idx_s] + wrapper_f + first_word + wrapper_b + sentence[first_idx_s:]
        first_idx_e += len(wrapper_f) + len(first_word) + len(wrapper_b)
        second_idx_s += len(wrapper_f) + len(first_word) + len(wrapper_b)
        second_idx_e += len(wrapper_f) + len(first_word) + len(wrapper_b)

        sentence = sentence[:first_idx_e+1] + wrapper_f + '/' + first_word + wrapper_b + sentence[first_idx_e+1:]
        second_idx_s += len(wrapper_f) + len('/') + len(first_word) + len(wrapper_b)
        second_idx_e += len(wrapper_f) + len('/') + len(first_word) + len(wrapper_b)

        sentence = sentence[:second_idx_s] + wrapper_f + second_word + wrapper_b + sentence[second_idx_s:]
        second_idx_e += len(wrapper_f) + len(second_word) + len(wrapper_b)

        sentence = sentence[:second_idx_e+1] + wrapper_f + '/' + second_word + wrapper_b + sentence[second_idx_e+1:]
        sentences_list.append(sentence)

    return np.array(sentences_list), entity_embedding_hint, tokenizer

def entity_marker_punct(df:pd.DataFrame, tokenizer) -> Tuple[np.ndarray, list, Callable]:
    sentences_list = []
    entity_embedding_hint = []
    for _, row in df.iterrows():
        subject_dict = literal_eval(row['subject_entity'])
        object_dict = literal_eval(row['object_entity'])
        sentence = row['sentence']
        if subject_dict['start_idx'] <= object_dict['start_idx']: # 만약 subject가 앞에 있을 경우
            first_idx_s = subject_dict['start_idx']
            first_idx_e = subject_dict['end_idx']
            second_idx_s = object_dict['start_idx']
            second_idx_e = object_dict['end_idx']
            first_word =  '@'
            second_word = '#'
        else:
            first_idx_s = object_dict['start_idx']
            first_idx_e = object_dict['end_idx']
            second_idx_s = subject_dict['start_idx']
            second_idx_e = subject_dict['end_idx']
            first_word = '#'
            second_word = '@'

        entity_embedding_hint.append((
            first_word + sentence[first_idx_s:first_idx_e+1] + first_word,
            second_word + sentence[second_idx_s:second_idx_e+1] + second_word
        ))

        sentence = sentence[:first_idx_s] + first_word + sentence[first_idx_s:]
        first_idx_e += len(first_word)
        second_idx_s += len(first_word)
        second_idx_e += len(first_word)

        sentence = sentence[:first_idx_e+1] + first_word + sentence[first_idx_e+1:]
        second_idx_s += len(first_word)
        second_idx_e += len(first_word)

        sentence = sentence[:second_idx_s] + second_word + sentence[second_idx_s:]
        second_idx_e += len(second_word)

        sentence = sentence[:second_idx_e+1] + second_word + sentence[second_idx_e+1:]
        sentences_list.append(sentence)
    
    return np.array(sentences_list), entity_embedding_hint, tokenizer

def typed_entity_marker_punct(df:pd.DataFrame, tokenizer) -> Tuple[np.ndarray, list, Callable]:
    sentences_list = []
    entity_embedding_hint = []
    for _, row in df.iterrows():
        subject_dict = literal_eval(row['subject_entity'])
        object_dict = literal_eval(row['object_entity'])
        sentence = row['sentence']
        if subject_dict['start_idx'] <= object_dict['start_idx']: # 만약 subject가 앞에 있을 경우
            first_idx_s = subject_dict['start_idx']
            first_idx_e = subject_dict['end_idx']
            second_idx_s = object_dict['start_idx']
            second_idx_e = object_dict['end_idx']
            first_word = '*' + subject_dict['type'] + '*'
            second_word = '∧' + object_dict['type'] + '∧'
            first_punc =  '@'
            second_punc = '#'
        else:
            first_idx_s = object_dict['start_idx']
            first_idx_e = object_dict['end_idx']
            second_idx_s = subject_dict['start_idx']
            second_idx_e = subject_dict['end_idx']
            first_word = '∧' + object_dict['type'] + '∧'
            second_word = '*' + subject_dict['type'] + '*'
            first_punc = '#'
            second_punc = '@'

        entity_embedding_hint.append((
            first_punc + first_word + sentence[first_idx_s:first_idx_e+1] + first_punc,
            second_punc + second_word + sentence[second_idx_s:second_idx_e+1] + second_punc
        ))

        sentence = sentence[:first_idx_s] + first_punc + first_word + sentence[first_idx_s:first_idx_e+1] + first_punc + sentence[first_idx_e+1:]

        first_idx_e += 2 * len(first_punc) + len(first_word)
        second_idx_s += 2 * len(first_punc) + len(first_word)
        second_idx_e += 2 * len(first_punc) + len(first_word)

        sentence = sentence[:second_idx_s] + second_punc + second_word + sentence[second_idx_s:second_idx_e+1] + second_punc + sentence[second_idx_e+1:]

        sentences_list.append(sentence)

    return np.array(sentences_list), entity_embedding_hint, tokenizer