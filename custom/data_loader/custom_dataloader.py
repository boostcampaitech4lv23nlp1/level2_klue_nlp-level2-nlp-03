import torch
import pandas as pd
from torch.utils.data import Dataset

class Customdataset(Dataset):
    def __init__(self, data, tokenizer, mode:bool):
        self.mode = mode
        self.df_data = self.load_data(data)
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        sentence = self.df_data.loc[idx, 'sentence']
        subject = self.df_data.loc[idx, 'subject_entity']
        object = self.df_data.loc[idx, 'object_entity']

        encoded_dict = self.tokenizer.encode_plus(
            sentence,
            subject+'[SEP]'+object,          
            add_special_tokens = True,      
            max_length = 128,
            padding='max_length',           
            # pad_to_max_length = True,
            truncation=True,
            return_attention_mask = True,   
            return_tensors = 'pt',        
            )
        
        if self.mode: #train, val
            label = self.df_data.loc[idx, 'label']
            return {'input_ids': encoded_dict.input_ids.squeeze(),
                    'attention_mask': encoded_dict.attention_mask.squeeze(), 
                    'labels': torch.tensor(label).to(torch.float).unsqueeze(dim=0)}
                    
        else: # test
            return {'input_ids': encoded_dict.input_ids.squeeze(),
                    'attention_mask': encoded_dict.attention_mask.squeeze(), 
                    }

    def __len__(self):
        return len(self.df_data)

    def load_data(self, data):
        dataset = data
        subject_entity = []
        object_entity = []
        num_label = []
        label_to_num = {}
        label_list = ['no_relation', 'org:top_members/employees', 'org:members',
            'org:product', 'per:title', 'org:alternate_names',
            'per:employee_of', 'org:place_of_headquarters', 'per:product',
            'org:number_of_employees/members', 'per:children',
            'per:place_of_residence', 'per:alternate_names',
            'per:other_family', 'per:colleagues', 'per:origin', 'per:siblings',
            'per:spouse', 'org:founded', 'org:political/religious_affiliation',
            'org:member_of', 'per:parents', 'org:dissolved',
            'per:schools_attended', 'per:date_of_death', 'per:date_of_birth',
            'per:place_of_birth', 'per:place_of_death', 'org:founded_by',
            'per:religion']

        for i in range(len(label_list)):
                label_to_num[label_list[i]] = i

        for i,j in zip(dataset['subject_entity'], dataset['object_entity']):
            i = i[1:-1].split(',')[0].split(':')[1]
            j = j[1:-1].split(',')[0].split(':')[1] 

            subject_entity.append(i)
            object_entity.append(j)
        out_dataset = pd.DataFrame({'sentence':dataset['sentence'],'subject_entity':subject_entity,'object_entity':object_entity})

        if self.mode:
            for i in dataset['label']:
                tmp = label_to_num[i]
                num_label.append(tmp)
            out_dataset['label'] = num_label
    
        self.num_to_label = label_list
        del subject_entity, object_entity, dataset, num_label, label_to_num
        return out_dataset