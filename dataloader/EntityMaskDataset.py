import torch
import numpy as np
from typing import Callable, Tuple
import pandas as pd
import einops as ein
from utils import pre_marker

class EntityMaskDataset(torch.utils.data.Dataset):
    """_summary_
    데이터를 불러와 전처리와 토크나이저 등 다양한 전처리를 수행하고
    data와 target을 나눠주는 작업을 해주는 클래스입니다.
    """
    def __init__(self, 
                 mode: str, # train / test 모드를 설정해줍니다.
                 data: pd.DataFrame, # 데이터셋을 불러올 root path를 지정해줍니다.
                 tokenizer: Callable,
                 entity_marker_mode:str=None,
                 max_length: int = 270, # 토크나이징할 문장의 최대 길이를 설정해줍니다.
                 ):
        super().__init__()
        self.mode = mode
        self.max_length = max_length
        self.entity_marker_mode = entity_marker_mode
        
        if self.mode == 'train':
            self.sentence_array, self.entity_hint, self.tokenizer, self.target_array = self._load_data(data, tokenizer)
        else:
            self.sentence_array, self.entity_hint, self.tokenizer = self._load_data(data, tokenizer)

    def _load_data(self, data:pd.DataFrame, tokenizer):
        """_summary_
        데이터 컬럼 : 'id', 'sentence', 'subject_entity', 'object_entity','label'
        그 중에 필요한 컬럼 : [features](str) : sentence_1, subject_entity, object_entity
                              [target](float) : label
        Returns:
            sentence(str)
            target(Optional[float])
        """
        # root path 안의 mode에 해당하는 csv 파일을 가져옵니다.
        sentence, entity_hint, tokenizer = getattr(pre_marker, self.entity_marker_mode)(data, tokenizer)
        if self.mode == 'train': # train or validation일 경우
            target = data['label'].to_numpy()
            
            return sentence, entity_hint, tokenizer, target
        else: # test일 경우
            return sentence, entity_hint, tokenizer

    # 입력하는 개수만큼 데이터를 사용합니다
    def __len__(self) -> int:
        return len(self.sentence_array)
    
    # 학습 및 추론 과정에서 데이터를 1개씩 꺼내오는 곳
    def __getitem__(self, idx):
        # 토크나이저 및 전처리를 위해 두 문장을 하나로 합쳐줍니다.
        sentence = self.sentence_array[idx]
        
        encoded_dict = self.tokenizer.encode_plus(
            sentence,    
            add_special_tokens = True,      
            max_length = self.max_length,           
            pad_to_max_length = True, # 여기서 이미 패딩을 수행합니다.
            truncation=True,
            return_attention_mask = True,   
            return_tensors = 'pt',
            )
        
        entity_mask1, entity_mask2 = self._entity_embedding(self.entity_hint[idx], encoded_dict['input_ids'][0])
        
        if self.mode == 'train':           
            return {'input_ids': ein.rearrange(encoded_dict.input_ids, '1 s -> s'),
                    'attention_mask': ein.rearrange(encoded_dict.attention_mask, '1 s -> s'), 
                    'labels': ein.rearrange(torch.tensor(self.target_array[idx], dtype=torch.long), ' -> 1'),
                    'entity_mask1' : entity_mask1,
                    'entity_mask2' : entity_mask2}

        else:            
            return {'input_ids': ein.rearrange(encoded_dict.input_ids, '1 s -> s'),
                    'attention_mask': ein.rearrange(encoded_dict.attention_mask, '1 s -> s'), 
                    'entity_mask1' : entity_mask1,
                    'entity_mask2' : entity_mask2}
    
    def _entity_embedding(self, entity_hint:torch.tensor, sentence:torch.tensor) ->  Tuple[torch.tensor, torch.tensor]:
        hint1 = self.tokenizer.encode(entity_hint[0], return_tensors='pt', add_special_tokens=False)[0]
        hint2 = self.tokenizer.encode(entity_hint[1], return_tensors='pt', add_special_tokens=False)[0]
        entity_embedding1 = []
        entity_embedding2 = []
        
        i = 0
        
        while sentence[i] != self.tokenizer.pad_token_id:
            if sentence[i] == hint1[0] and not entity_embedding1: # 임베딩을 찾지 못한 경우만 실행
                if torch.equal(sentence[i:i+len(hint1)], hint1): # hint1 길이만큼 슬라이싱한 뒤 같은지 비교
                    entity_embedding1 = [x for x in range(i,i+len(hint1))]
                    i += len(hint1) -1

            elif sentence[i] == hint2[0] and not entity_embedding2:
                if torch.equal(sentence[i:i+len(hint2)], hint2):
                    entity_embedding2 = [x for x in range(i,i+len(hint2))]
                    i += len(hint2) -1
            i += 1
            if entity_embedding1 and entity_embedding2:
                break # 임베딩을 모두 찾은 경우 바로 종료
            
        # [0, 0, 0, 1, 1, 1, 0, 0, 0] 형태로 만들어줌
        first = [0] * self.max_length
        second = [0] * self.max_length
        for i in entity_embedding1:
            first[i] = 1
        for i in entity_embedding2:
            second[i] = 1
                
        return torch.tensor(first, dtype=torch.long), torch.tensor(second, dtype=torch.long)