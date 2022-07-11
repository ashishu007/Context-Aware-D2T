import json
from tqdm import tqdm
from datasets import load_dataset
from extract_ents import ExtractEntities

class TeamStreak:
    def __init__(self, ftr_type='text') -> None:
        self.ftr_type = ftr_type
        self.ee_obj = ExtractEntities()
        self.dataset = load_dataset('GEM/sportsett_basketball')

    def team_streak_text_ftr(self, streak, broken_streak, team_name):
        """
        text: Philladelphia 76ers have won 1 consecutive matches. They broke 5 matches loosing streak.
        """
        text = f"{team_name} have {'won' if streak['type'] == 'W' else 'lost'} {streak['count']} consecutive matches."
        if broken_streak:
            text += f" They broke {broken_streak['count']} matches {'loosing' if broken_streak['type'] == 'L' else 'winning'} streak."
        return text

    def team_streak_num_ftr(self, streak, broken_streak):
        ids = {'W': 1, 'L': 2}
        ftrs = [streak['count'], ids[streak['type']]]
        if 'type' in broken_streak:
            ftrs.append(ids[broken_streak['type']])
        else:
            ftrs.append(0)
        if 'count' in broken_streak:
            ftrs.append(broken_streak['count'])
        else:
            ftrs.append(0)
        return ftrs

    def aug_text_ftr(self, streak, broken_streak, team_name):
        streak1 = {'type': streak['type'], 'count': streak['count'] + 1}
        broken_streak1 = {'type': broken_streak['type'], 'count': broken_streak['count'] + 1} if broken_streak else {}
        aug_text1 = self.team_streak_text_ftr(streak1, broken_streak1, team_name)
        streak2 = {'type': streak['type'], 'count': streak['count'] - 1}
        broken_streak2 = {'type': broken_streak['type'], 'count': broken_streak['count'] - 1} if broken_streak else {}
        aug_text2 = self.team_streak_text_ftr(streak2, broken_streak2, team_name)
        return [aug_text1, aug_text2]

    def get_theme_train_val_test_data(self):
        all_data = {'train': [], 'validation': [], 'test': []}
        augmented_train_texts = []

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
                    hftrs = self.team_streak_text_ftr(hstreak, hbroken_streak, hteam)
                elif self.ftr_type == 'num':
                    hftrs = self.team_streak_num_ftr(hstreak, hbroken_streak)

                if self.ftr_type == 'text' and part == 'train':
                    augmented_train_texts.extend(self.aug_text_ftr(hstreak, hbroken_streak, hteam))

                vstreak = streak_info[gem_id]['vis']
                vbroken_streak = {}
                if gem_id in broken_streaks_info:
                    if 'vis' in broken_streaks_info[gem_id]:
                        vbroken_streak = broken_streaks_info[gem_id]['vis']
                if self.ftr_type == 'text':
                    vftrs = self.team_streak_text_ftr(vstreak, vbroken_streak, vteam)
                elif self.ftr_type == 'num':
                    vftrs = self.team_streak_num_ftr(vstreak, vbroken_streak)

                if self.ftr_type == 'text' and part == 'train':
                    augmented_train_texts.extend(self.aug_text_ftr(vstreak, vbroken_streak, vteam))

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

        return all_data, augmented_train_texts


class TeamStanding:
    def __init__(self, ftr_type='text') -> None:
        self.train_mentions = json.load(open(f'standings/mentions_train.json'))
        self.train_info = json.load(open(f'standings/train.json'))

        self.valid_mentions = json.load(open(f'standings/mentions_validation.json'))
        self.valid_info = json.load(open(f'standings/validation.json'))

        self.test_mentions = json.load(open(f'standings/mentions_test.json'))
        self.test_info = json.load(open(f'standings/test.json'))

        self.ftr_type = ftr_type

    def team_standing_text_ftr(self, info):
        """
        current_text: Boston Celtics are at place 4 in the standings with 10 wins, 5 losses and win percent of 66%. 
        prev_1_text: The team above them is Brooklyn Nets at place 3 in the stadings with 11 wins, 5 losses and win percent of 68%.
        next_1_text: The team below them is Toronto Raptors at place 5 in the standing with 9 wins, 5 losses and win percent of 64%.
        season_date_text: This is the 49 day of the season.
        """

        current_text = f'{info["current"]["team"]} are at place {info["current"]["standing"]} in the standings with {info["current"]["wins"]} wins, {info["current"]["losses"]} losses and win percent of {info["current"]["win_perct"]}%.'
        season_date_text = f'This is the {info["current"]["season_date"]} day of the season.'

        if 'prev_1' in info:
            prev_1_text = f'The team above them is {info["prev_1"]["team"]} at place {info["prev_1"]["standing"]} in the stadings with {info["prev_1"]["wins"]} wins, {info["prev_1"]["losses"]} losses and win percent of {info["prev_1"]["win_perct"]}%.'
        else:
            prev_1_text = 'There is no team above them.'

        if 'next_1' in info:
            next_1_text = f'The team below them is {info["next_1"]["team"]} at place {info["next_1"]["standing"]} in the standing with {info["next_1"]["wins"]} wins, {info["next_1"]["losses"]} losses and win percent of {info["next_1"]["win_perct"]}%.'
        else:
            next_1_text = 'There is no team below them.'

        return f'{current_text} {prev_1_text} {next_1_text} {season_date_text}'

    def team_standing_num_ftr(self, info):
        ftr_vector = {}
        current_team_ftrs = ['standing', 'wins', 'losses', 'win_perct', 'season_date']
        other_team_ftrs = ['standing', 'wins', 'losses', 'win_perct', 'win_diff']

        for key, val in info['current'].items():
            if key in current_team_ftrs:
                ftr_vector[f"current-{key}"] = val

        if 'prev_1' in info:
            for key in other_team_ftrs:
                ftr_vector[f"prev_1-{key}"] = info['prev_1'][key]
        else:
            for key in other_team_ftrs:
                ftr_vector[f"prev_1-{key}"] = 0

        if 'next_1' in info:
            for key in other_team_ftrs:
                ftr_vector[f"next_1-{key}"] = info['next_1'][key]
        else:
            for key in other_team_ftrs:
                ftr_vector[f"next_1-{key}"] = 0

        return list(ftr_vector.values())

    def aug_text_ftr(self, info):
        info1 = info.copy()
        info2 = info.copy()
        info1['current']['standing'] = info['current']['standing'] + 1
        info2['current']['standing'] = info['current']['standing'] - 1
        if 'next_1' in info1:
            info1['next_1']['standing'] = info1['next_1']['standing'] + 1
        if 'prev_1' in info1:
            info1['prev_1']['standing'] = info1['prev_1']['standing'] + 1
        if 'next_1' in info2:
            info2['next_1']['standing'] = info2['next_1']['standing'] - 1
        if 'prev_1' in info2:
            info2['prev_1']['standing'] = info2['prev_1']['standing'] - 1
        return [self.team_standing_text_ftr(info1), self.team_standing_text_ftr(info2)]

    def get_theme_train_val_test_data(self):
        all_data = {'train': [], 'validation': [], 'test': []}
        augmented_train_texts = []
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
                    ftrs.append(self.team_standing_text_ftr(inf['home']))
                    ftrs.append(self.team_standing_text_ftr(inf['vis']))
                    if part == 'train':
                        augmented_train_texts.extend(self.aug_text_ftr(inf['home']))
                        augmented_train_texts.extend(self.aug_text_ftr(inf['vis']))
                elif self.ftr_type == 'num':
                    ftrs.append(self.team_standing_num_ftr(inf['home']))
                    ftrs.append(self.team_standing_num_ftr(inf['vis']))

                lab = 1 if gem_id in mentions else 0
                labs.append(lab)
                labs.append(lab)

            all_data[part] = {'ftrs': ftrs, 'labs': labs}
        return all_data, augmented_train_texts
