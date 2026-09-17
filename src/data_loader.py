import spacy
nlp = spacy.load("en_core_web_sm")


import copy
import pickle
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass, field


@dataclass
class Loader:
    dir_workspace : str
    version_exp : str
    data_name : str
    N_icl : list = field(default_factory=lambda: [1,3,5,10])
    flag_replace_NER : bool = False
    # limit to the first n_users (sorted) per pickle; None means all users
    n_users : int = None

    def __post_init__(self):
        self.dir_data = f"{self.dir_workspace}/preprocessed_data/{self.version_exp}/{self.data_name}"
        self._load_item()

    # filter dict to first n_users by sorted key order
    def _subset_users(self, d: dict) -> dict:
        if self.n_users is None:
            return d
        keys = sorted(d.keys())[:self.n_users]
        return {k: d[k] for k in keys}

    # apply _subset_users to each inner dict independently
    def _subset_users_dd(self, dd: dict) -> dict:
        return {k: self._subset_users(d) for k, d in dd.items()}

    def _load_item(self):
        dd_text = dict()
        for item_type in ["candidates", "history"]:
            df_i = pd.read_csv(f"{self.dir_data}/items_{item_type}.csv", index_col=0).fillna("")

            # for NER removal processing
            ## genres with capital cases tend to be treated as NER by spacy
            if "ARD" in self.data_name:
                a = "category"
            elif self.data_name == "MovieLens":
                a = "genres"
            else:
                a = "categories"

            try:
                df_i[a] = df_i[a].apply(lambda s : s.lower())
            except Exception:
                pass

            # texualize
            di = {
                item : "\n".join([f"{k} : {v}" for k,v in d_.items()])
                for item, d_ in df_i.T.to_dict().items()
            }
            dd_text[item_type] = di
        self.dd_item = copy.deepcopy(dd_text)
        return self

    def _list_profiles(self):
        if self.data_name == "MovieLens":
            profiles = ["profile"]
        elif self.data_name == "Job":
            profiles = [f"profile_{f}" for f in ["mid-career", "new-graduate"]]
        else:
            profiles = []
        return profiles

    def profile(self):
        dd_text = dict()
        for profile in self._list_profiles():
            with open(f"{self.dir_data}/records_{profile}.pickle", 'rb') as f:
                d_data = pickle.load(f)
            # subset users
            d_data = self._subset_users(d_data)
            # texualize
            dd_text[profile] = {user : str(d["profile"]) for user, d in d_data.items()}
        return dd_text

    def flag(self):
        dict_flag = dict()
        for text_type in self._list_profiles() + self._list_icl():
            with open(f"{self.dir_data}/records_{text_type}.pickle", 'rb') as f:
                d_data = pickle.load(f)
            # subset users
            d_data = self._subset_users(d_data)
            d_flag = dict()
            for user, du in d_data.items():
                items_pos = du["candidates_positive"].split(", ")
                items_neg = sorted(du["candidates_negative"].split(", "))  # fix sampling order by sort item names
                d_flag[user] = pd.Series(
                    [1]*len(items_pos) + [0]*len(items_neg),
                    index=items_pos+items_neg
                ).to_dict()
            dict_flag[text_type] = copy.deepcopy(d_flag)
        return dict_flag

    def _list_icl(self):
        return [f"{n_icl}-sample" for n_icl in self.N_icl]

    def interaction(self):
        dd_text = dict()
        for user_type in self._list_icl():
            with open(f"{self.dir_data}/records_{user_type}.pickle", 'rb') as f:
                d_data = pickle.load(f)
            # subset users
            d_data = self._subset_users(d_data)
            dd_text[user_type] = {
                user : list(du["history"].values()) for user, du in d_data.items()
            }
        return dd_text

    def to_NER(self, d):
        def _replace_ner_with_tags(text):
            doc = nlp(text)
            out = []
            last = 0
            for ent in doc.ents:
                out.append(text[last:ent.start_char])
                out.append(f"<{ent.label_}>")
                last = ent.end_char
            out.append(text[last:])
            return "".join(out)

        d_ = {k : _replace_ner_with_tags(str(v)) for k,v in tqdm(d.items(), desc="to NER")}
        return copy.deepcopy(d_)

    def _ner(self):
        ner = "with_NER" if self.flag_replace_NER else "original"
        return ner

    def _change_ner(self, d_, name):
        path_ner = f"{self.dir_data}/text_{name}_{self._ner()}.pickle"
        try:
            with open(path_ner, 'rb') as f:
                d_ner = pickle.load(f)
        except Exception:
            if self.flag_replace_NER:
                d_ner = self.to_NER(d_)
            else:
                d_ner = copy.deepcopy(d_)
            with open(path_ner, 'wb') as f:
                pickle.dump(d_ner, f)
        return d_ner

    def change_ner(self, dd):
        return {k : self._change_ner(v, k) for k,v in dd.items()}

    def load_data(self):
        dict_data = {
            "items" : self.change_ner(self.dd_item),
            # re-apply subset after change_ner to discard cached full-user data
            "profile" : self._subset_users_dd(self.change_ner(self.profile())),
            "interactions" : self.interaction(),
            "flag" : self.flag(),
            "ner" : self._ner()
        }
        for text_type in ["concat", "separate"]:
            dict_data[text_type] = self.texualize(dict_data, text_type)
        return dict_data

    def texualize(self, dict_data, text_type):
        d_item_history = dict_data["items"]["history"]
        def _to_text(items, text_type="separate"):
            if text_type == "separate":
                return [d_item_history[item] for item in items]
            else:
                return "\n".join([f"#log {idx+1}\n" + d_item_history[item] for idx, item in enumerate(items)])

        def to_text(d, text_type="separate"):
            return {user : _to_text(items, text_type=text_type) for user, items in d.items()}

        dict_ = {f"{text_type}_{k}" : to_text(d, text_type) for k,d in dict_data["interactions"].items()}
        return dict_
