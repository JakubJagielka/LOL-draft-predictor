import json
import csv
from riotwatcher import LolWatcher, ApiError
from multiprocessing import Process
import logging
import time

logging.basicConfig(level=logging.INFO)
def fetch_match(match_id, region,lol_watcher):
    try:
        match_details = lol_watcher.match.by_id(region=region, match_id=match_id)
        return match_details
    except ApiError as e:
        logging.warning(f"API Error: {e}")
        time.sleep(1)
        return None

def append_to_csv(file_name, games_data, region):
    with open(file_name, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for game in games_data:
            writer.writerow([
                game['match_id'],
                game['winning_team'],
                game['gameDuration'],
                game['championId'],
                game['kda'],
                game['kills'],
                game['assists'],
                game['deaths'],
                game['goldPerMinute'],
                game['damagePerMinute'],
                game['enemyChampionImmobilizations'],
                game['immobilizeAndKillWithAlly'],
                game['killParticipation'],
                game['laneMinionsFirst10Minutes'],
                game['teamDamagePercentage'],
                game['turretPlatesTaken'],
                game['turretTakedowns'],
                game['champExperience'],
                game['damageDealtToBuildings'],
                game['damageSelfMitigated'],
                game['goldEarned'],
                game['totalDamageDealtToChampions'],
                game['totalDamageShieldedOnTeammates'],
                game['totalDamageTaken'],
                game['totalHealsOnTeammates'],
                game['totalMinionsKilled'],
                game['visionScore'],
                game['dragonTakedowns'],
            ])

        with open('games_data.'+region+'.csv', mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            for game in games_data:
                writer.writerow([game['match_id'], game['championId'], game['winning_team']])


def fetch_and_process_games(puuids, region, file_name, api_key):
    lol_watcher = LolWatcher(api_key)  # Create a LolWatcher instance
    unique_matches = set()
    with open('games_data.'+region+'.csv', 'r', newline='') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            match_id = row['Match ID']
            unique_matches.add(match_id)


    for puuid in puuids:
        with open('current_puuid' + region + '.txt', 'w') as f:
            f.write(puuid)
        try:
            match_ids = lol_watcher.match.matchlist_by_puuid(region=region, puuid=puuid, count=100)

            matches = list(unique_matches)
            games_data = []
            unique_elements = [element for element in match_ids if element not in unique_matches]
            unique_matches.update(unique_elements)


            for match_id in unique_elements:
                match_details = fetch_match(match_id, api_key, region,lol_watcher)
                try:
                    # i need to get all the data from match_details, the data which i need looks like : ['info']: ['gameDuration'] ['info']['participants'][0-9]: ['assists'] ['challenges']:['damagePerMinute'] ['enemyChampionImmobilizations'] ['goldPerMinute'] ['immobilizeAndKillWithAlly'] ['kda'] ['killParticipation'] ['laneMinionsFirst10Minutes'] ['teamDamagePercentage'] ['turretPlatesTaken'] ['turretTakedowns'] ['info']['participants'][0-9]: ['champExperience'] ['championId'] ['damageDealtToBuildings'] ['damageSelfMitigated'] ['deaths'] ['goldEarned'] ['kills'] ['totalDamageDealtToChampions'] ['totalDamageShieldedOnTeammates'] ['totalDamageTaken'] ['totalHealsOnTeammates'] ['totalMinionsKilled'] ['visionScore'] ['dragonTakedowns']
                    if match_details and len(match_details['info']['participants']) == 10:
                        games_data.append({
                            'match_id': match_id,
                            'winning_team': match_details['info']['teams'][0]['win'],
                            'gameDuration': match_details['info']['gameDuration'],
                            'championId': [participant['championId'] for participant in match_details['info']['participants']],
                            'kda': [participant['challenges']['kda'] for participant in match_details['info']['participants']],
                            'kills': [participant['kills'] for participant in match_details['info']['participants']],
                            'assists': [participant['assists'] for participant in match_details['info']['participants']],
                            'deaths': [participant['deaths'] for participant in match_details['info']['participants']],
                            'goldPerMinute': [participant['challenges']['goldPerMinute'] for participant in match_details['info']['participants']],
                            'damagePerMinute': [participant['challenges']['damagePerMinute'] for participant in match_details['info']['participants']],
                            'enemyChampionImmobilizations': [participant['challenges']['enemyChampionImmobilizations'] for participant in match_details['info']['participants']],
                            'immobilizeAndKillWithAlly': [participant['challenges']['immobilizeAndKillWithAlly'] for participant in match_details['info']['participants']],
                            'killParticipation': [participant['challenges']['killParticipation'] for participant in match_details['info']['participants']],
                            'laneMinionsFirst10Minutes': [participant['challenges']['laneMinionsFirst10Minutes'] for participant in match_details['info']['participants']],
                            'teamDamagePercentage': [participant['challenges']['teamDamagePercentage'] for participant in match_details['info']['participants']],
                            'turretPlatesTaken': [participant['challenges']['turretPlatesTaken'] for participant in match_details['info']['participants']],
                            'turretTakedowns': [participant['challenges']['turretTakedowns'] for participant in match_details['info']['participants']],
                            'champExperience': [participant['champExperience'] for participant in match_details['info']['participants']],
                            'damageDealtToBuildings': [participant['damageDealtToBuildings'] for participant in match_details['info']['participants']],
                            'damageSelfMitigated': [participant['damageSelfMitigated'] for participant in match_details['info']['participants']],
                            'goldEarned': [participant['goldEarned'] for participant in match_details['info']['participants']],
                            'totalDamageDealtToChampions': [participant['totalDamageDealtToChampions'] for participant in match_details['info']['participants']],
                            'totalDamageShieldedOnTeammates': [participant['totalDamageShieldedOnTeammates'] for participant in match_details['info']['participants']],
                            'totalDamageTaken': [participant['totalDamageTaken'] for participant in match_details['info']['participants']],
                            'totalHealsOnTeammates': [participant['totalHealsOnTeammates'] for participant in match_details['info']['participants']],
                            'totalMinionsKilled': [participant['totalMinionsKilled'] for participant in match_details['info']['participants']],
                            'visionScore': [participant['visionScore'] for participant in match_details['info']['participants']],
                            'dragonTakedowns': [participant['challenges']['dragonTakedowns'] for participant in match_details['info']['participants']],
                        })
                except KeyError as e:
                    logging.error(f"Key error: {e}")
                    continue


            append_to_csv(file_name, games_data, region)
            logging.info(f"Saved {len(games_data)} games to {file_name}.")
        except ApiError as e:
            logging.warning(f"API Error: {e}")
            continue

if __name__ == '__main__':
    api_key = 'RGAPI-0c3cdf66-061a-46ff-bfd1-639c57ca6907'
    puuids = json.load(open('puuideuw1.json', 'r'))
    puuids_kr = json.load(open('puuidkr.json', 'r'))

    file_name = 'games_data_test' + 'euw1' + '.csv'
    file_namekr = 'games_data_test'+ 'kr' + '.csv'

    with open(file_name, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Match ID', 'Winning Team','gameDuration','championId','kda','kills','assists','deaths','goldPerMinute','damagePerMinute','enemyChampionImmobilizations','immobilizeAndKillWithAlly','killParticipation','laneMinionsFirst10Minutes','teamDamagePercentage','turretPlatesTaken','turretTakedowns','champExperience','damageDealtToBuildings','damageSelfMitigated','goldEarned','totalDamageDealtToChampions','totalDamageShieldedOnTeammates','totalDamageTaken','totalHealsOnTeammates','totalMinionsKilled','visionScore','dragonTakedowns'])


    with open(file_namekr, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Match ID', 'Winning Team','gameDuration','championId','kda','kills','assists','deaths','goldPerMinute','damagePerMinute','enemyChampionImmobilizations','immobilizeAndKillWithAlly','killParticipation','laneMinionsFirst10Minutes','teamDamagePercentage','turretPlatesTaken','turretTakedowns','champExperience','damageDealtToBuildings','damageSelfMitigated','goldEarned','totalDamageDealtToChampions','totalDamageShieldedOnTeammates','totalDamageTaken','totalHealsOnTeammates','totalMinionsKilled','visionScore','dragonTakedowns'])

    p1 = Process(target=fetch_and_process_games, args=(puuids,'euw1',file_name,api_key))
    p2= Process(target=fetch_and_process_games, args=(puuids_kr, 'kr', file_namekr, api_key) )

    p1.start()
    p2.start()