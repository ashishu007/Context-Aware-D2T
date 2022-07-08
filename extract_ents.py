class ExtractEntities:
    alias_dict = {
        'Mavs': 'Mavericks',
        'Cavs': 'Cavaliers',
        'Sixers': '76ers',
    }

    prons = set(["he", "He", "him", "Him", "his", "His", "they", "They", "them", "Them", "their", "Their"]) # leave out "it"
    singular_prons = set(["he", "He", "him", "Him", "his", "His"])
    plural_prons = set(["they", "They", "them", "Them", "their", "Their"])

    number_words = set(["one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                        "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
                        "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty",
                        "sixty", "seventy", "eighty", "ninety", "hundred", "thousand"])
    days_set = set(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

    ordinal_set = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9,
                        "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15,
                        "1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5, "6th": 6, "7th": 7, "8th": 8, "9th": 9, "10th": 10, 
                        "11th": 11, "12th": 12, "13th": 13, "14th": 14, "15th": 15}

    def get_full_team_ents(self, team_ents_in_sent, item):
        teams_mentioned = [] #set()
        vname, vplace = item['teams']['vis']['name'], item['teams']['vis']['place']
        hname, hplace = item['teams']['home']['name'], item['teams']['home']['place']
        hnon, hnop = item['teams']['home']['next_game']['opponent_name'], item['teams']['home']['next_game']['opponent_place']
        vnon, vnop = item['teams']['vis']['next_game']['opponent_name'], item['teams']['vis']['next_game']['opponent_place']
        hng_ents = [hnon, hnop, f"{hnop} {hnon}"]
        vng_ents = [vnon, vnop, f"{vnop} {vnon}"]
        home_team = [hname, hplace, f'{hplace} {hname}']
        vis_team = [vname, vplace, f'{vplace} {vname}']
        home_flag = False
        vis_flag = False
        hng_flag = False
        vng_flag = False
        for ent in team_ents_in_sent:
            if ent in home_team:
                home_flag = True
            if ent in vis_team:
                vis_flag = True
            if ent in hng_ents:
                hng_flag = True
            if ent in vng_ents:
                vng_flag = True
        if home_flag:
            teams_mentioned.append(f'{hplace} {hname}')
        if vis_flag:
            teams_mentioned.append(f'{vplace} {vname}')
        if hng_flag:
            teams_mentioned.append(f"{hnop} {hnon}")
        if vng_flag:
            teams_mentioned.append(f"{vnop} {vnon}")
        teams_mentioned_final = []
        for ent in teams_mentioned:
            if ent not in teams_mentioned_final:
                teams_mentioned_final.append(ent)
        return teams_mentioned_final

    def get_full_player_ents(self, player_ents_in_sent, item):

        hbs = item['teams']['home']['box_score']
        vbs = item['teams']['vis']['box_score']

        home_full_names = [player['name'] for player in hbs]
        home_first_names = [player['first_name'] for player in hbs]
        home_last_names = [player['last_name'] for player in hbs]

        vis_full_names = [player['name'] for player in vbs]
        vis_first_names = [player['first_name'] for player in vbs]
        vis_last_names = [player['last_name'] for player in vbs]

        home_player_mention_idxs, vis_player_mention_idxs = [], []

        for player_ent in player_ents_in_sent:
            home_player_idx = -1
            vis_player_idx = -1
            if player_ent in home_full_names:
                for idx2, player in enumerate(hbs):
                    if player['name'] == player_ent:
                        home_player_idx = idx2
            elif player_ent in home_first_names:
                for idx2, player in enumerate(hbs):
                    if player['first_name'] == player_ent:
                        home_player_idx = idx2
            elif player_ent in home_last_names:
                for idx2, player in enumerate(hbs):
                    if player['last_name'] == player_ent:
                        home_player_idx = idx2

            if player_ent in vis_full_names:
                for idx2, player in enumerate(vbs):
                    if player['name'] == player_ent:
                        vis_player_idx = idx2
            elif player_ent in vis_first_names:
                for idx2, player in enumerate(vbs):
                    if player['first_name'] == player_ent:
                        vis_player_idx = idx2
            elif player_ent in vis_last_names:
                for idx2, player in enumerate(vbs):
                    if player['last_name'] == player_ent:
                        vis_player_idx = idx2

            if home_player_idx != -1:
                home_player_mention_idxs.append(home_player_idx)
            if vis_player_idx != -1:
                vis_player_mention_idxs.append(vis_player_idx)

        full_player_ents = []
        for i in home_player_mention_idxs:
            full_player_ents.append(hbs[i]['name'])
        for i in vis_player_mention_idxs:
            full_player_ents.append(vbs[i]['name'])

        players_final = []
        for ent in full_player_ents:
            if ent not in players_final:
                players_final.append(ent)
        return players_final

    def get_all_ents(self, score_dict):
        players = []#set()
        teams = []#set()

        teams.append(score_dict['teams']['home']['name'])
        teams.append(score_dict['teams']['vis']['name'])
        
        teams.append(score_dict['teams']['home']['place'])
        teams.append(score_dict['teams']['vis']['place'])
        
        teams.append(f"{score_dict['teams']['home']['place']} {score_dict['teams']['home']['name']}")
        teams.append(f"{score_dict['teams']['vis']['place']} {score_dict['teams']['vis']['name']}")

        teams.append(score_dict['teams']['home']['next_game']['opponent_name'])
        teams.append(score_dict['teams']['vis']['next_game']['opponent_name'])

        teams.append(score_dict['teams']['home']['next_game']['opponent_place'])
        teams.append(score_dict['teams']['vis']['next_game']['opponent_place'])

        teams.append(f"{score_dict['teams']['home']['next_game']['opponent_place']} {score_dict['teams']['home']['next_game']['opponent_name']}")
        teams.append(f"{score_dict['teams']['vis']['next_game']['opponent_place']} {score_dict['teams']['vis']['next_game']['opponent_name']}")

        for player in score_dict['teams']['home']['box_score']:
            players.append(player['first_name'])
            players.append(player['last_name'])
            players.append(player['name'])

        for player in score_dict['teams']['vis']['box_score']:
            players.append(player['first_name'])
            players.append(player['last_name'])
            players.append(player['name'])

        teams_final, players_final = [], []
        for ent in teams:
            if ent not in teams_final:
                teams_final.append(ent)
        for ent in players:
            if ent not in players_final:
                players_final.append(ent)
        
        all_ents = teams + players

        return all_ents, teams, players

    def extract_entities(self, all_ents, sent):

        new_toks = []
        for tok in sent.split(' '):
            if tok in self.alias_dict:
                new_toks.append(self.alias_dict[tok])
            else:
                new_toks.append(tok)
        new_sent = ' '.join(new_toks)

        toks = new_sent.split(' ')
        sent_ents = []
        i = 0
        while i < len(toks):
            if toks[i] in all_ents:
                j = 1
                while i+j <= len(toks) and " ".join(toks[i:i+j]) in all_ents:
                    j += 1
                sent_ents.append(" ".join(toks[i:i+j-1]))
                i += j-1
            else:
                i += 1
        sent_ents_final = []
        for ent in sent_ents:
            if ent not in sent_ents_final:
                sent_ents_final.append(ent)
        return sent_ents_final

    def extract_entities_ie(self, sent, all_ents, prons, prev_ents=None, resolve_prons=False,
            players=None, teams=None, cities=None):
        sent_ents = []
        i = 0
        ents_list = list(all_ents)
        while i < len(sent):
            if sent[i] in prons:
                if resolve_prons:
                    referent = self. deterministic_resolve(sent[i], players, teams, cities, sent_ents, prev_ents)
                    if referent is None:
                        sent_ents.append((i, i+1, sent[i], True)) # is a pronoun
                    else:
                        #print "replacing", sent[i], "with", referent[2], "in", " ".join(sent)
                        sent_ents.append((i, i+1, referent[2], False)) # pretend it's not a pron and put in matching string
                else:
                    sent_ents.append((i, i+1, sent[i], True)) # is a pronoun
                i += 1
            elif sent[i] in all_ents: # findest longest spans; only works if we put in words...
                j = 1
                while i+j <= len(sent) and " ".join(sent[i:i+j]) in all_ents:
                    # print("i:{} j:{} sent[i:i+j]={} ent_i:{} ent_i_j:{}".format(i,j,sent[i:i+j],ents_list[ents_list.index(sent[i])], ents_list[ents_list.index(" ".join(sent[i:i+j]))]))
                    j += 1
                sent_ents.append((i, i+j-1, " ".join(sent[i:i+j-1]), False))
                i += j-1
            else:
                i += 1
        return sent_ents

    def deterministic_resolve(self, pron, players, teams, cities, curr_ents, prev_ents, max_back=1):
        # we'll just take closest compatible one.
        # first look in current sentence; if there's an antecedent here return None, since
        # we'll catch it anyway
        for j in range(len(curr_ents)-1, -1, -1):
            if pron in self.singular_prons and curr_ents[j][2] in players:
                return None
            elif pron in self.plural_prons and curr_ents[j][2] in teams:
                return None
            elif pron in self.plural_prons and curr_ents[j][2] in cities:
                return None

        # then look in previous max_back sentences
        if len(prev_ents) > 0:
            for i in range(len(prev_ents)-1, len(prev_ents)-1-max_back, -1):
                for j in range(len(prev_ents[i])-1, -1, -1):
                    if pron in self.singular_prons and prev_ents[i][j][2] in players:
                        return prev_ents[i][j]
                    elif pron in self.plural_prons and prev_ents[i][j][2] in teams:
                        return prev_ents[i][j]
                    elif pron in self.plural_prons and prev_ents[i][j][2] in cities:
                        return prev_ents[i][j]
        return None
