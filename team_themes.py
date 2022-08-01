import json
from tqdm import tqdm
from datasets import load_dataset
from extract_ents import ExtractEntities

class TeamStreak:
    def __init__(self, ftr_type='num') -> None:
        self.ftr_type = ftr_type
        self.ee_obj = ExtractEntities()
        self.dataset = load_dataset('GEM/sportsett_basketball')
        self.team_names = json.load(open('data/all_teams.json'))
        self.team_popularity = json.load(open('data/teams_popularity.json'))

    def team_streak_text_ftr(self, streak, broken_streak, team_name):
        """
        text: Philladelphia 76ers have won 1 consecutive matches. They broke 5 matches loosing streak.
        """
        text = f"{team_name} have {'won' if streak['type'] == 'W' else 'lost'} {streak['count']} consecutive matches."
        if broken_streak:
            text += f" They broke {broken_streak['count']} matches {'loosing' if broken_streak['type'] == 'L' else 'winning'} streak."
        return text

    def team_streak_num_ftr(self, streak, broken_streak, team_name, line_score):
        """
        ftrs: [10, 1, 1, 2, 5]
        ftrs[0]: team id
        ftrs[1]: streak count
        ftrs[2]: streak type
        ftrs[3]: broken streak type
        ftrs[4]: broken streak count
        cat_ftrs = [0, 2, 3]
        """
        ids = {'W': 1, 'L': 2}
        team_id = self.team_names[team_name]
        # ftrs = [team_id, streak['count'], ids[streak['type']]]
        team_pop = self.team_popularity[team_name] if team_name in self.team_popularity else 0
        ftrs = {"team_name": team_id, "team_popularity": team_pop, "streak_count": streak['count'], "streak_type": ids[streak['type']]}
        if 'type' in broken_streak:
            # ftrs.append(ids[broken_streak['type']])
            ftrs['broken_streak_type'] = ids[broken_streak['type']]
        else:
            # ftrs.append(0)
            ftrs['broken_streak_type'] = 0
        if 'count' in broken_streak:
            # ftrs.append(broken_streak['count'])
            ftrs['broken_streak_count'] = broken_streak['count']
        else:
            # ftrs.append(0)
            ftrs['broken_streak_count'] = 0
        # ftrs.extend(list(line_score.values()))
        return ftrs

    def aug_train_ftrs(self, streak, broken_streak, team_name, line_score, ftr_type='text'):
        streak1 = {'type': streak['type'], 'count': streak['count'] + 1}
        broken_streak1 = {'type': broken_streak['type'], 'count': broken_streak['count'] + 1} if broken_streak else {}
        if ftr_type == 'text':
            aug_ftrs1 = self.team_streak_text_ftr(streak1, broken_streak1, team_name)
        elif ftr_type == 'num':
            aug_ftrs1 = self.team_streak_num_ftr(streak1, broken_streak1, team_name, line_score)
        streak2 = {'type': streak['type'], 'count': streak['count'] - 1}
        broken_streak2 = {'type': broken_streak['type'], 'count': broken_streak['count'] - 1} if broken_streak else {}
        if ftr_type == 'text':
            aug_ftrs2 = self.team_streak_text_ftr(streak2, broken_streak2, team_name)
        elif ftr_type == 'num':
            aug_ftrs2 = self.team_streak_num_ftr(streak2, broken_streak2, team_name, line_score)
        return [aug_ftrs1, aug_ftrs2]

    def get_theme_train_val_test_data(self):
        # all_data = {'train': [], 'validation': [], 'test': []}
        all_data = {}
        augmented_train_ftrs = []

        for part in ['train',  'validation', 'test']:
            print(f'processing {part}...')

            part_data = {'ftrs': [], 'labs': []}
            js = json.load(open(f'data/{part}_data_ct.json'))
            streak_info = json.load(open(f'streaks/{part}.json'))
            broken_streaks_info = json.load(open(f'streaks/broken_streak-{part}.json'))

            for item in tqdm(self.dataset[f'{part}']):
                gem_id_data_points = []
                gem_id = item['gem_id']
                hls = item['teams']['home']['line_score']['game']
                vls = item['teams']['vis']['line_score']['game']
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
                    hftrs = self.team_streak_num_ftr(hstreak, hbroken_streak, hteam, hls)

                # if part == 'train':
                #     augmented_train_ftrs.extend(self.aug_train_ftrs(hstreak, hbroken_streak, hteam, hls, ftr_type=self.ftr_type))

                vstreak = streak_info[gem_id]['vis']
                vbroken_streak = {}
                if gem_id in broken_streaks_info:
                    if 'vis' in broken_streaks_info[gem_id]:
                        vbroken_streak = broken_streaks_info[gem_id]['vis']
                if self.ftr_type == 'text':
                    vftrs = self.team_streak_text_ftr(vstreak, vbroken_streak, vteam)
                elif self.ftr_type == 'num':
                    vftrs = self.team_streak_num_ftr(vstreak, vbroken_streak, vteam, vls)

                # if part == 'train':
                #     augmented_train_ftrs.extend(self.aug_train_ftrs(vstreak, vbroken_streak, vteam, vls, ftr_type=self.ftr_type))

                if htflag == True:
                    # part_data['ftrs'].append(hftrs)
                    # part_data['labs'].append(1)
                    gem_id_data_points.append({'ftrs': hftrs, 'lab': 1})
                elif htflag == False:
                    # part_data['ftrs'].append(hftrs)
                    # part_data['labs'].append(0)
                    gem_id_data_points.append({'ftrs': hftrs, 'lab': 0})

                if vtflag == True:
                    # part_data['ftrs'].append(vftrs)
                    # part_data['labs'].append(1)
                    gem_id_data_points.append({'ftrs': vftrs, 'lab': 1})
                elif vtflag == False:
                    # part_data['ftrs'].append(vftrs)
                    # part_data['labs'].append(0)
                    gem_id_data_points.append({'ftrs': vftrs, 'lab': 0})

                all_data[gem_id] = gem_id_data_points

            # all_data[part] = part_data

        return all_data, augmented_train_ftrs


class TeamStanding:
    def __init__(self, ftr_type='num') -> None:
        self.ftr_type = ftr_type
        self.ee_obj = ExtractEntities()
        self.dataset = load_dataset('GEM/sportsett_basketball')
        self.team_names = json.load(open('data/all_teams.json'))
        self.team_popularity = json.load(open('data/teams_popularity.json'))

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

    def team_standing_num_ftr(self, info, line_score):
        """
        ftrs[0], ftrs[1], ftrs[2], ftrs[3], ftrs[4], ftrs[5]: team_name, current_standing, current_wins, current_losses, current_win_perct, current_season_date
        ftrs[6], ftrs[7], ftrs[8], ftrs[9], ftrs[10]: prev_1_standing, prev_1_wins, prev_1_losses, prev_1_win_perct, prev_1_season_date
        ftrs[11], ftrs[12], ftrs[13], ftrs[14], ftrs[15]: next_1_standing, next_1_wins, next_1_losses, next_1_win_perct, next_1_season_date
        cat_ftrs = [0]
        """
        team_name = self.team_names[info["current"]["team"]]
        team_pop = self.team_popularity[team_name] if team_name in self.team_popularity else 0
        ftr_vector = {'team_name': team_name, 'team_pop': team_pop}

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

        return ftr_vector
        # return list(ftr_vector.values())
        # return list(ftr_vector.values()) + list(line_score.values())

    def aug_train_ftrs(self, info, ftr_type='text'):
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
        if ftr_type == 'text':
            return [self.team_standing_text_ftr(info1), self.team_standing_text_ftr(info2)]
        elif ftr_type == 'num':
            return [self.team_standing_num_ftr(info1), self.team_standing_num_ftr(info2)]

    def get_theme_train_val_test_data(self):
        # all_data = {'train': [], 'validation': [], 'test': []}
        # all_data = {'train': {}, 'validation': {}, 'test': {}}
        all_data = {}
        augmented_train_ftrs = []

        for part in ['train', 'validation', 'test']:
            print(f'Processing {part}...')

            # if part != 'train':
            #     continue

            js = json.load(open(f'data/{part}_data_ct.json'))
            info = json.load(open(f'standings/{part}.json'))
            ftrs, labs = [], []
            # tr_ftrs, tr_labs = [], []
            # vl_ftrs, vl_labs = [], []
            # te_ftrs, te_labs = [], []

            for item in tqdm(self.dataset[f'{part}']):
                gem_id = item['gem_id']
                season = item['game']['season']
                all_ents, teams, players = self.ee_obj.get_all_ents(item)
                hteam = f"{item['teams']['home']['place']} {item['teams']['home']['name']}"
                vteam = f"{item['teams']['vis']['place']} {item['teams']['vis']['name']}"
                hls = item['teams']['home']['line_score']['game']
                vls = item['teams']['vis']['line_score']['game']
                item_sents = list(filter(lambda x: x['gem_id'] == gem_id, js))
                hflag, vflag = False, False
                uids = set([x['summary_idx'] for x in item_sents])
                for uid in uids:
                    sents = list(filter(lambda x: x['summary_idx'] == uid, item_sents))
                    for sent in sents:
                        ner_sent = sent['ner_abs_sent']
                        coref_sent = sent['coref_sent']
                        team_ents_in_sent1 = self.ee_obj.extract_entities(teams, coref_sent)
                        team_ents_in_sent = self.ee_obj.get_full_team_ents(team_ents_in_sent1, item)
                        if len(team_ents_in_sent) == 0:
                            continue
                        if 'seed' in ner_sent or 'ORDINAL - place' in ner_sent or 'ORDINAL place' in ner_sent:
                            if hteam in team_ents_in_sent:
                                hflag = True
                                break
                            if vteam in team_ents_in_sent:
                                vflag = True
                                break

                hinf, vinf = info[gem_id]['home'], info[gem_id]['vis']
                if self.ftr_type == 'num':
                    hftr_vector = self.team_standing_num_ftr(hinf, hls)
                    vftr_vector = self.team_standing_num_ftr(vinf, vls)
                elif self.ftr_type == 'text':
                    hftr_vector = self.team_standing_text_ftr(hinf)
                    vftr_vector = self.team_standing_text_ftr(vinf)

                # if part == 'train':
                #     hftr_vector1, vftr_vector1 = self.aug_train_ftrs(hinf, ftr_type=self.ftr_type), self.aug_train_ftrs(vinf, ftr_type=self.ftr_type)
                #     augmented_train_ftrs.append(hftr_vector1)
                #     augmented_train_ftrs.append(vftr_vector1)
                
                gem_id_data_points = []

                if hflag:
                    # # if season == '2014':
                    # #     tr_ftrs.append(hftr_vector)
                    # #     tr_labs.append(1)
                    # # elif season == '2015':
                    # #     vl_ftrs.append(hftr_vector)
                    # #     vl_labs.append(1)
                    # # elif season == '2016':
                    # #     te_ftrs.append(hftr_vector)
                    # #     te_labs.append(1)
                    # ftrs.append(hftr_vector)
                    # labs.append(1)
                    gem_id_data_points.append({'ftrs': hftr_vector, 'lab': 1})
                else:
                    # # if season == '2014':
                    # #     tr_ftrs.append(hftr_vector)
                    # #     tr_labs.append(0)
                    # # elif season == '2015':
                    # #     vl_ftrs.append(hftr_vector)
                    # #     vl_labs.append(0)
                    # # elif season == '2016':
                    # #     te_ftrs.append(hftr_vector)
                    # #     te_labs.append(0)
                    # ftrs.append(hftr_vector)
                    # labs.append(0)
                    gem_id_data_points.append({'ftrs': hftr_vector, 'lab': 0})

                if vflag:
                    # # if season == '2014':
                    # #     tr_ftrs.append(vftr_vector)
                    # #     tr_labs.append(1)
                    # # elif season == '2015':
                    # #     vl_ftrs.append(vftr_vector)
                    # #     vl_labs.append(1)
                    # # elif season == '2016':
                    # #     te_ftrs.append(vftr_vector)
                    # #     te_labs.append(1)
                    # ftrs.append(vftr_vector)
                    # labs.append(1)
                    gem_id_data_points.append({'ftrs': vftr_vector, 'lab': 1})
                else:
                    # # if season == '2014':
                    # #     tr_ftrs.append(vftr_vector)
                    # #     tr_labs.append(0)
                    # # elif season == '2015':
                    # #     vl_ftrs.append(vftr_vector)
                    # #     vl_labs.append(0)
                    # # elif season == '2016':
                    # #     te_ftrs.append(vftr_vector)
                    # #     te_labs.append(0)
                    # ftrs.append(vftr_vector)
                    # labs.append(0)
                    gem_id_data_points.append({'ftrs': vftr_vector, 'lab': 0})
                
                # all_data[part].update({gem_id: gem_id_data_points})
                all_data[gem_id] = gem_id_data_points

            # all_data[part] = {'ftrs': ftrs, 'labs': labs}
            # # all_data['train'] = {'ftrs': tr_ftrs, 'labs': tr_labs}
            # # all_data['validation'] = {'ftrs': vl_ftrs, 'labs': vl_labs}
            # # all_data['test'] = {'ftrs': te_ftrs, 'labs': te_labs}
        return all_data, augmented_train_ftrs
