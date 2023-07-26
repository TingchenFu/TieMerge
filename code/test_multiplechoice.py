import argparse
from datasets import load_dataset
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer,T5Config
from transformers import default_data_collator
from str2bool import str2bool
from tqdm import tqdm
# how to write parallel evaluation 


# class SimpleBatcher:
#     def __init__(self, tokenizer,max_prompt_length,max_target_length,device) -> None:
#         self.max_prompt_length = max_prompt_length
#         self.max_target_length = max_target_length
#         self.tokenizer=tokenizer
#         self.device=device

#     def promptppl(self,prompt_list,tgt_list,architecture):
#         assert architecture in ['encoder-decoder','decoder-only','target-only']
#         if architecture=='decoder-only':
#             batch_input_id=[]
#             batch_label=[]
#             bs = len(prompt_list)
#             for i in range(bs):
#                 src=self.tokenizer.encode(prompt_list[i],add_special_tokens=False, padding=False,truncation=True, max_length=self.max_prompt_length)
#                 tgt = self.tokenizer.encode(tgt_list[i], padding=False,truncation=True,max_length=self.max_target_length)
#                 input_id=src+tgt
#                 label = [-100]*len(src) + tgt
#                 batch_input_id.append(input_id)
#                 batch_label.append(label)
            
#             max_length=max([len(x) for x in batch_input_id])
#             for i in range(bs):
#                 padding_length = max_length - len(batch_input_id[i])
#                 if padding_length:
#                     batch_input_id[i].extend([self.tokenizer.pad_token_id] * padding_length)
#                     batch_label[i].extend([-100]*padding_length)
#                 assert len(batch_input_id[i]) == len(batch_label[i]) == max_length

#             batch_input_id=torch.tensor(batch_input_id,dtype=torch.long,device=self.device)
#             attention_mask=(batch_input_id!=self.tokenizer.pad_token_id).to(self.device)

#             return{
#                 'input_id':batch_input_id,
#                 'attention_mask':attention_mask,
#                 'label':torch.tensor(batch_label,dtype=torch.long,device=self.device)
#             }
        
#         elif architecture=='encoder-decoder':
#             tokenized_source=self.tokenizer.batch_encode_plus(prompt_list,max_length=self.max_prompt_length,truncation=True,padding='longest',return_tensors='pt')
#             tokenized_target=self.tokenizer.batch_encode_plus(tgt_list,max_length=self.max_target_length,truncation=True,padding='longest',return_tensors='pt')
#             label=[
#                 [(l if l!= self.tokenizer.pad_token_id else -100) for l in label] for label in tokenized_target['input_ids']
#             ]
#             return {
#                 'input_id':tokenized_source['input_ids'].to(self.device),
#                 'attention_mask':tokenized_source['attention_mask'].to(self.device),
#                 'label':torch.tensor(label,dtype=torch.long,device=self.device)
#             }
#         else:
#             raise NotImplementedError
        





if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name",type=str,)
    parser.add_argument("--model_name_or_path",type=str,default=None)
    parser.add_argument("--config_name",type=str)
    parser.add_argument("--tokenizer_name",type=str)
    parser.add_argument("--test_file",type=str,default=None)

    parser.add_argument("--bs",type=int,default=4)
    parser.add_argument("--length_norm",type=str2bool,default=False)
    parser.add_argument("--max_input_length",type=int,default=128)
    parser.add_argument("--max_target_length",type=int,default=8)
    
    args = parser.parse_args()
    
    config = T5Config.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
    )
    tokenizer = T5Tokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    )
    model = T5ForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    def template_function(example,template):
        templated = template.apply(example)
        example['input'] = templated[0]
        example['choice']=template.get_answer_choices_list(example)
        example['answer']= example['choice'].index(templated[1]) 
        return example

    def tokenize_function(examples,max_input_length,max_target_length):
        n_choice= len(examples['choice'][0])
        # print(len(examples['choice']))
        choice_id=[]
        choice_mask=[]
        for choice in examples['choice']:
            assert len(choice) == n_choice
            choice_encoded = tokenizer.batch_encode_plus(choice,truncation=True, max_length= max_target_length,padding='max_length')
            # print(len(choice_id))
            choice_id.append(choice_encoded['input_ids'])
            choice_mask.append(choice_encoded['attention_mask'])

        batch_encode= tokenizer.batch_encode_plus(examples['input'],truncation=True,max_length = max_input_length,padding='max_length')
        # print(n_choice)
        # print(len(choice_id))
        #print(len(batch_encode['input_ids']))
        assert all([len(x) ==n_choice  for x in choice_id])
        #assert  len(examples['input'])   * n_choice == len(choice_id)
        batch_encode['choice_id'] = choice_id
        batch_encode['choice_mask'] = choice_mask
        return batch_encode
    
    templates=[]
    from promptsource.templates import DatasetTemplates
    for template in DatasetTemplates(args.dataset_name).templates.values():
        ignore = False
        if not template.metadata.original_task:
            ignore = True
        for metric in template.metadata.metrics:
            if metric not in ['Accuracy']:
                ignore = True
        if not ignore:
            templates.append(template)


    test_dataset = load_dataset("json",data_files=args.test_file,split='train')
    test_dataset = test_dataset.map(
        template_function,
        batched=False,
        fn_kwargs={'template':templates[0]},
        remove_columns=test_dataset.column_names
    )
    test_dataset = test_dataset.map(
        tokenize_function,
        batched=True,
        fn_kwargs={'max_input_length':args.max_input_length,'max_target_length':args.max_target_length},
        remove_columns=['input','choice']
    )
    test_dataloader = DataLoader(
            test_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.bs,pin_memory=True,
        )
    print(test_dataset.column_names) 

    count=0
    hit=0
    progress_bar= tqdm(range(len(test_dataloader)))
    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            encoder_hidden = model.encoder(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],return_dict=True)['last_hidden_state']
            bs= batch['input_ids'].shape[0]
            
            choice_id = batch['choice_id']
            choice_mask =batch['choice_mask']
            choice_id = choice_id.reshape(bs, -1, args.max_target_length)
            n_choice = choice_id.shape[1]


            # convert the 0-th dim from bs to bs*n_choice
            attention_mask = torch.repeat_interleave(batch['attention_mask'], n_choice, dim=0)
            encoder_hidden = (
                torch.repeat_interleave(encoder_hidden, n_choice, dim=0),
            )

            transformer_outputs =  model(
                attention_mask=attention_mask,
                encoder_outputs=encoder_hidden,
                labels=choice_id.reshape(bs*n_choice, args.max_target_length),
                return_dict=True
            )

            # [batch_size x num_choices, max_choice_len, vocab_size]
            #(bs*n_choice, max_choice_len, vocab_size)
            choice_logit = transformer_outputs[1].float()
            max_choice_len = choice_logit.shape[1]
            # [batch_size x num_choices x max_choice_len]
            choice_logprob = -F.cross_entropy(
                choice_logit.view(-1, choice_logit.shape[-1]),
                choice_id.view(-1),
                reduction="none",
            )

            choice_logprob = choice_logprob.reshape(
                -1,  n_choice, max_choice_len
            )
            choice_mask = choice_mask.reshape(-1, n_choice, max_choice_len)
            #allChoices_masks = allChoices_masks.reshape(-1, num_choices, maxChoice_len)
        # Zero out padded out tokens so we their log probability is not included
            choice_logprob = choice_logprob * choice_mask

            #(bs,n_choice)
            choice_logprob = torch.sum(choice_logprob, dim=2)

            #(bs,n_choice)
            choice_len = torch.sum(choice_mask, dim=2)

            if args.length_norm:
                choice_logprob = choice_logprob/ choice_len
            #(bs)
            choice_pred = torch.argmax(choice_logprob,dim=1)
            count += choice_pred.shape[0]
            hit += (choice_pred==batch['answer']).sum().item()
            progress_bar.update(1)


    print("hit :{}".format(hit))
    print("n_example: {} ".format(count))

