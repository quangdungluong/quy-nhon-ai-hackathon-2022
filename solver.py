from app_model import create_model, create_tokenizer
import torch
from vncorenlp import VnCoreNLP
from utils import get_final_prediction, seed_everything

class ClassifyReviewSolver:
    def __init__(self, config):
        # setting_seed(1)
        self.config = config
        self.model = create_model(self.config.model_name, self.config.model_type)
        self.model.load_state_dict(torch.load(self.config.model_ckpt, map_location=self.config.device))
        self.model = self.model.to(self.config.device)
        self.model = self.model.eval()
        self.tokenizer = create_tokenizer(self.config.model_name)
        self.rdrsegmenter = VnCoreNLP(self.config.rdrsegmenter_path, annotators="wseg", max_heap_size='-Xmx500m')
        # self.replace_dict = {"Nvien" : "nhân viên", "Nv": "nhân viên", "NV":"nhân viên", "nvien": "nhân viên", "nv":"nhân viên", "NVien":"nhân viên",
        #                     "ks":"khách sạn", "ksan":"khách sạn", "Ks":"khách sạn", "KS":"khách sạn", "Ksan":"khách sạn"}
    def solve(self, text):
        segmented_text = ' '.join([' '.join(sent) for sent in self.rdrsegmenter.tokenize(text)])
        encoding = self.tokenizer(segmented_text, max_length=self.config.max_len,
                                  padding='max_length',
                                  add_special_tokens=True,
                                  truncation=True)
        encoding['input_ids'] = torch.tensor(encoding['input_ids']).flatten()
        encoding['attention_mask'] = torch.tensor(encoding['attention_mask']).flatten()
        
        encoding['input_ids'] = encoding['input_ids'][None, :]
        encoding['attention_mask'] = encoding['attention_mask'][None, :]
        
        outputs = self.model(input_ids=encoding['input_ids'], attention_mask=encoding['attention_mask'])
        logits = []
        for out in outputs.detach().numpy():
            logits.append(out)
        predictions = get_final_prediction(logits, 0.5)
        predictions = predictions[0][0].tolist()
        print(predictions)
        return predictions