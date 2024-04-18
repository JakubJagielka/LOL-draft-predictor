from riotwatcher import LolWatcher
import json



def get_top_players(region):
    '''
    uses riotwatcher API to retrieve players in challenger, GM, and masters.
    returns a list of all summoner IDs
    '''

    #return challengers
    challengers = lol_watcher.league.challenger_by_queue(region, 'RANKED_SOLO_5x5')

    #return grandmasters
    gms = lol_watcher.league.grandmaster_by_queue(region, 'RANKED_SOLO_5x5')

    #return masters
    masters = lol_watcher.league.masters_by_queue(region, 'RANKED_SOLO_5x5')

    #list of the above objects
    all_top_players = [challengers, gms, masters]

    #loop through and concat all summoner Ids
    summoner_ids = []
    for division in all_top_players:
        for entry in division['entries']:
            summoner_ids.append(list(entry.values())[0])

    return summoner_ids

def get_puuid(summoner_ids, region):
    '''
    take in a summoner ID from riot API and fetches the users puuid.
    this is done because other queries require the puuid.
    returns dict object mapping summoner id to puuid.
    '''

    summid_to_puuid = []
    path = 'puuid' + region + '.json'
    for summoner in summoner_ids:
        try:
            s = lol_watcher.summoner.by_id(region, summoner)['puuid']
            #in my puuids.json file i have list and i wanna append each s to it each time i loop through
            summid_to_puuid.append(s)
            with open(path,'w') as f:
                json.dump(summid_to_puuid, f)
        except:
            print('error')
            continue

    return summid_to_puuid


if __name__ == '__main__':
    api = ''
    lol_watcher = LolWatcher(api)
    eu = get_top_players('euw1')
    kr = get_top_players('kr')

    with open('puuid_euw1.json', 'w') as f:
        json.dump(get_puuid(eu, api, 'euw1'), f)

    with open('puuid_kr.json', 'w') as f:
        json.dump(get_puuid(kr, api, 'kr'), f)