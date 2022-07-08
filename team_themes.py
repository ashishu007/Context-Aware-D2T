import json
from tqdm import tqdm
from datasets import load_dataset
from utils import FeatureExtractor
from extract_ents import ExtractEntities

class TeamStreak:
    def __init__(self, ftr_type='text') -> None:
        self.ftr_type = ftr_type
        self.ee_obj = ExtractEntities()
        self.ftr_ext = FeatureExtractor()
        self.dataset = load_dataset('GEM/sportsett_basketball')

    def get_theme_train_val_test_data(self):
        all_data = {'train': [], 'validation': [], 'test': []}

        for part in ['train',  'validation', 'test']:
            print(f'processing {part}...')

            part_data = {'ftrs': [], 'labs': []}
            js = json.load(open(f'data/{part}_data_ct.json'))
            streak_info = json.load(open(f'streaks/{part}.json'))
            broken_streaks_info = json.load(open(f'streaks/broken_streak-{part}.json'))

            for item in tqdm(self.dataset[f'{part}']):
                gem_id = item['gem_id']
                all_ents, teams, players = self.ee_obj.get_all_ents(item)
                hteam = f"{item['teams']['home']['place']} {item['teams']['home']['name']}"
                vteam = f"{item['teams']['vis']['place']} {item['teams']['vis']['name']}"
                summary_sents = list(filter(lambda x: x['gem_id'] == item['gem_id'], js))
                uids = set([x['summary_idx'] for x in summary_sents])
                htflag, vtflag = False, False
                for uid in uids:
                    uid_sents = list(filter(lambda x: x['summary_idx'] == uid, summary_sents))
                    for sent in uid_sents:
                        sentence = sent['coref_sent']
                        teams_in_sent1 = self.ee_obj.extract_entities(teams, sentence)
                        teams_in_sent = self.ee_obj.get_full_team_ents(teams_in_sent1, item)
                        if len(teams_in_sent) == 0:
                            continue
                        if 'streak' in sentence or 'straight' in sentence:
                            if hteam in teams_in_sent:
                                htflag = True
                            if vteam in teams_in_sent:
                                vtflag = True

                hstreak = streak_info[gem_id]['home']
                hbroken_streak = {}
                if gem_id in broken_streaks_info:
                    if 'home' in broken_streaks_info[gem_id]:
                        hbroken_streak = broken_streaks_info[gem_id]['home']
                if self.ftr_type == 'text':
                    hftrs = self.ftr_ext.team_streak_text_ftr(hstreak, hbroken_streak, hteam)
                elif self.ftr_type == 'num':
                    hftrs = self.ftr_ext.team_streak_num_ftr(hstreak, hbroken_streak)

                vstreak = streak_info[gem_id]['vis']
                vbroken_streak = {}
                if gem_id in broken_streaks_info:
                    if 'vis' in broken_streaks_info[gem_id]:
                        vbroken_streak = broken_streaks_info[gem_id]['vis']
                if self.ftr_type == 'text':
                    vftrs = self.ftr_ext.team_streak_text_ftr(vstreak, vbroken_streak, vteam)
                elif self.ftr_type == 'num':
                    vftrs = self.ftr_ext.team_streak_num_ftr(vstreak, vbroken_streak)

                if htflag == True:
                    part_data['ftrs'].append(hftrs)
                    part_data['labs'].append(1)
                elif htflag == False:
                    part_data['ftrs'].append(vftrs)
                    part_data['labs'].append(0)
                
                if vtflag == True:
                    part_data['ftrs'].append(vftrs)
                    part_data['labs'].append(1)
                elif vtflag == False:
                    part_data['ftrs'].append(hftrs)
                    part_data['labs'].append(0)

            all_data[part] = part_data
    
        return all_data
    
class TeamStanding:
    def __init__(self, ftr_type='text') -> None:
        self.train_mentions = json.load(open(f'standings/mentions_train.json'))
        self.train_info = json.load(open(f'standings/train.json'))

        self.valid_mentions = json.load(open(f'standings/mentions_validation.json'))
        self.valid_info = json.load(open(f'standings/validation.json'))

        self.test_mentions = json.load(open(f'standings/mentions_test.json'))
        self.test_info = json.load(open(f'standings/test.json'))

        self.ftr_type = ftr_type
        self.ftr_ext = FeatureExtractor()
    
    def get_theme_train_val_test_data(self):
        all_data = {'train': [], 'validation': [], 'test': []}
        for part in ['train', 'validation', 'test']:
            if part == 'train':
                mentions = self.train_mentions
                info = self.train_info
            elif part == 'validation':
                mentions = self.valid_mentions
                info = self.valid_info
            elif part == 'test':
                mentions = self.test_mentions
                info = self.test_info

            print(f'Processing {part}...')
            print(f'{len(mentions)} mentions')
            print(f'{len(info)} info')
            ftrs, labs = [], []
            for gem_id, inf in tqdm(info.items()):
                if self.ftr_type == 'text':
                    ftrs.append(self.ftr_ext.team_standing_text_ftr(inf['home']))
                    ftrs.append(self.ftr_ext.team_standing_text_ftr(inf['vis']))
                elif self.ftr_type == 'num':
                    ftrs.append(self.ftr_ext.team_standing_num_ftr(inf['home']))
                    ftrs.append(self.ftr_ext.team_standing_num_ftr(inf['vis']))

                lab = 1 if gem_id in mentions else 0
                labs.append(lab)
                labs.append(lab)

            all_data[part] = {'ftrs': ftrs, 'labs': labs}
        return all_data