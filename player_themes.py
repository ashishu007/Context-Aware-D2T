import re
import json
from tqdm import tqdm
from datasets import load_dataset
from extract_ents import ExtractEntities
from text2num import text2num, NumberException

class PlayerAverage:
    def __init__(self, ftr_type='num') -> None:
        self.ftr_type = ftr_type
        self.ee_obj = ExtractEntities()
        self.dataset = load_dataset('GEM/sportsett_basketball')
        self.player_names = json.load(open('data/all_players.json'))

    def replace_num_tokens(self, sent):
        new_toks = []
        for toke in sent.split():
            if toke == '-' or 'previous' in toke:
                continue
            try:
                num = text2num(toke)
                new_toks.append(str(num))
            except:
                new_toks.append(toke)
        return ' '.join(new_toks)

    def get_y_from_aver_pattern(self, sent_new):
        """
        get Y from the sents like: player A is averaging X points in the last Y games
        """
        stats = {}
        if len(re.findall(r'\d+\s+g[a-z]+\s+', sent_new)) != 0 or len(re.findall(r'\d+\s+o[a-z]+\s+', sent_new)) != 0 or len(re.findall(r'\d+\s+start[a-z]+\s+', sent_new)) != 0:        
            try:
                finds = re.findall(r'\d+\s+g[a-z]+\s+', sent_new)[0].strip().split(' ')[0]
                stats['Y'] = int(finds)
            except:
                pass

            try:
                finds = re.findall(r'\d+\s+out[a-z]+\s+', sent_new)[0].strip().split(' ')[0]
                stats['Y'] = int(finds)
            except:
                pass

            try:
                finds = re.findall(r'\d+\s+start[a-z]+\s+', sent_new)[0].strip().split(' ')[0]
                stats['Y'] = int(finds)
            except:
                pass

        elif 'season' in sent_new or 'year' in sent_new:
            stats['Y'] = 100
        
        if 'Y' in stats:
            pts = re.findall(r'\d+\s+point[a-z]+\s+', sent_new)
            reb = re.findall(r'\d+\s+rebound[a-z]+\s+', sent_new)
            ast = re.findall(r'\d+\s+assist[a-z]+\s+', sent_new)
            stl = re.findall(r'\d+\s+steal[a-z]+\s+', sent_new)
            blk = re.findall(r'\d+\s+block[a-z]+\s+', sent_new)
            if len(pts) != 0:
                stats['PTS'] = int(pts[0].strip().split(' ')[0])
            if len(reb) != 0:
                stats['TREB'] = int(reb[0].strip().split(' ')[0])
            if len(ast) != 0:
                stats['AST'] = int(ast[0].strip().split(' ')[0])
            if len(stl) != 0:
                stats['STL'] = int(stl[0].strip().split(' ')[0])
            if len(blk) != 0:
                stats['BLK'] = int(blk[0].strip().split(' ')[0])

        return stats

    def player_average_num_ftr(self, player_name, player_hist_info, phkeys_id):
        player_id = self.player_names[player_name] if player_name in self.player_names else 0
        clf_ftrs = []
        for k, v in player_hist_info.items():
            if k == 'name':
                continue
            temp = k.split('_')
            f0 = phkeys_id[temp[0]]
            f1 = 100 if temp[-1] == 'season' else int(temp[-2])
            f2 = v
            f3 = 1 if temp[1] == 'avg' else 2
            clf_ftrs.append([player_id, f0, f1, f2, f3])
        return clf_ftrs

    def player_average_text_ftr(self, player_name, player_hist_info, phkeys_id):
        """
        text: 
        """
        text = f"{player_name} is averaging/totaling X points/rebounds/assists/steals/blocks in the last Y games"
        pass

    def get_theme_train_val_test_data(self):
        all_data = {'train': [], 'validation': [], 'test': []}
        augmented_train_ftrs = []

        for part in ['train',  'validation', 'test']:
            print(f'processing {part}')
            js = json.load(open(f'data/{part}_data_ct.json'))
            hist_info = json.load(open(f'average_stats/{part}.json'))

            clf_samples = {'ftrs': [], 'labs': []}
            phkeys_id = {'PTS': 1, 'AST': 2, 'TREB': 3, 'BLK': 4, 'STL': 5}

            for item in tqdm(self.dataset[f'{part}']):
                gem_id = item['gem_id']
                all_ents, teams, players = self.ee_obj.get_all_ents(item)
                hplayers = [player['name'] for player in item['teams']['home']['box_score']]
                vplayers = [player['name'] for player in item['teams']['vis']['box_score']]
                summary_sents = list(filter(lambda x: x['gem_id'] == item['gem_id'], js))
                uids = set([x['summary_idx'] for x in summary_sents])
                for uid in uids:
                    uid_sents = list(filter(lambda x: x['summary_idx'] == uid, summary_sents))
                    for sent in uid_sents:
                        sentence = sent['coref_sent']
                        players_in_sent1 = self.ee_obj.extract_entities(players, sentence)
                        players_in_sent = self.ee_obj.get_full_player_ents(players_in_sent1, item)
                        if len(players_in_sent) == 0:
                            continue
                        new_sentence = self.replace_num_tokens(sentence)

                        if 'avera' in new_sentence or 'total' in new_sentence:
                            player_name = players_in_sent[0]
                            player_this_game_team_side = 'home' if player_name in hplayers else 'vis'
                            player_hist_info = hist_info[gem_id][player_this_game_team_side][player_name]
                            player_hist_clf_ftrs = self.player_average_num_ftr(player_name, player_hist_info, phkeys_id)
                            hist_stats_from_sent = self.get_y_from_aver_pattern(new_sentence)

                            if 'Y' not in hist_stats_from_sent:
                                continue

                            for clf_ftr in player_hist_clf_ftrs:
                                if clf_ftr[-3] == 100:
                                    for k, v in player_hist_info.items():
                                        if k == 'name':
                                            continue
                                        if 'season' in k:
                                            clf_samples['ftrs'].append(clf_ftr)
                                            clf_samples['labs'].append(1)
                                        else:
                                            clf_samples['ftrs'].append(clf_ftr)
                                            clf_samples['labs'].append(0)
                                elif clf_ftr[-3] > 0 and clf_ftr[-3] < 100:
                                    for k, v in player_hist_info.items():
                                        if k == 'name' or 'season' in k:
                                            continue
                                        if 'avg' in k and int(k.split('_')[-2]) == clf_ftr[-3] and 'avera' in new_sentence:
                                            clf_samples['ftrs'].append(clf_ftr)
                                            clf_samples['labs'].append(1)
                                        elif 'total' in k and int(k.split('_')[-2]) == clf_ftr[-3] and 'total' in new_sentence:
                                            clf_samples['ftrs'].append(clf_ftr)
                                            clf_samples['labs'].append(1)
                                        else:
                                            clf_samples['ftrs'].append(clf_ftr)
                                            clf_samples['labs'].append(0)
            all_data[part] = clf_samples

        return all_data, augmented_train_ftrs


class PlayerDouble:
    def __init__(self, ftr_type='num') -> None:
        self.ftr_type = ftr_type
        self.ee_obj = ExtractEntities()
        self.dataset = load_dataset('GEM/sportsett_basketball')
        self.player_names = json.load(open('data/all_players.json'))

    def player_double_num_ftr(self, doubles_info, player_name):
        pcode = self.player_names[player_name] if player_name in self.player_names else 0
        clf_ftrs = [pcode]
        for key, val in doubles_info.items():
            clf_ftrs.append(val)
        return clf_ftrs

    def get_theme_train_val_test_data(self):

        all_data = {'train': [], 'validation': [], 'test': []}
        augmented_train_ftrs = []

        for part in ['train',  'validation', 'test']:

            print(f'processing {part}')
            part_data = {'ftrs': [], 'labs': []}
            js = json.load(open(f'data/{part}_data_ct.json'))
            doubles_info = json.load(open(f'doubles/{part}.json'))

            for item in tqdm(self.dataset[f'{part}']):
                gem_id = item['gem_id']
                all_ents, teams, players = self.ee_obj.get_all_ents(item)
                hplayers = [player['name'] for player in item['teams']['home']['box_score']]
                vplayers = [player['name'] for player in item['teams']['vis']['box_score']]
                summary_sents = list(filter(lambda x: x['gem_id'] == item['gem_id'], js))
                uids = set([x['summary_idx'] for x in summary_sents])
                for uid in uids:
                    uid_sents = list(filter(lambda x: x['summary_idx'] == uid, summary_sents))
                    for sent in uid_sents:
                        sentence = sent['coref_sent']
                        players_in_sent1 = self.ee_obj.extract_entities(players, sentence)
                        players_in_sent = self.ee_obj.get_full_player_ents(players_in_sent1, item)
                        if len(players_in_sent) == 0:
                            continue

                        player_in_this_sent = players_in_sent[0]
                        player_in_this_sent_side = 'home' if player_in_this_sent in hplayers else 'vis'
                        player_double_hist_info = doubles_info[gem_id][player_in_this_sent_side][player_in_this_sent]
                        clf_ftrs = self.player_double_num_ftr(player_double_hist_info, player_in_this_sent)
                        label = 0
                        if '- double' in sentence:
                            if 'straight' in sentence or 'row' in sentence or 'total' in sentence:
                                label = 1
                        part_data['ftrs'].append(clf_ftrs)
                        part_data['labs'].append(label)

            all_data[part] = part_data
        
        return all_data, augmented_train_ftrs
