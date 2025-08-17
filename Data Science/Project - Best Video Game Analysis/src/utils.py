import requests
import pandas as pd
import time

def scrape_steam_data():
    """Scrape data from Steam API and Steam Spy.

    Returns:
        pd.DataFrame: A CSV DataFrame file 'steam_games_combined.csv' containing combined data from Steam API and Steam Spy.
    """
    
    print("Starting to scrape Steam data")
    # Step 1: Get all Steam apps
    def get_steam_apps():
        steam_apps = []
        count = 0
        
        response = requests.get("https://api.steampowered.com/ISteamApps/GetAppList/v2/")
        apps = response.json()['applist']['apps']
        for app in apps:
            response = requests.get(f"https://store.steampowered.com/api/appdetails?appids={app['appid']}").json()
            app_data = response.get(str(app['appid']), {})
            if app_data.get('success') and app_data['data'].get('type') == 'game':
                data = app_data['data']
                steam_apps.append(data)
                count += 1
                if count % 100 == 0:
                    print(f"Processed {count} Steam apps")
                time.sleep(0.1) # small delay to avoid rate limits
        return pd.DataFrame(steam_apps)

    steam_df = get_steam_apps()
    steam_df.rename(columns={'steam_appid': 'appid'}, inplace=True)
    print(f"Steam apps DataFrame generation finished")

    # Step 2: Function to get Steam Spy data
    def get_steamspy_data(appid):
        url = f"https://steamspy.com/api.php?request=appdetails&appid={appid}"
        try:
            resp = requests.get(url, timeout=5)
            data = resp.json()
            if 'appid' in data:
                return data
            else:
                return None
        except:
            return None

    # Step 3: Loop over apps and collect Steam Spy data (only real games)
    steamspy_list = []
    count = 0

    for appid in steam_df['appid']:
        data = get_steamspy_data(appid)
        if data:
            steamspy_list.append(data)
        count += 1
        if count % 100 == 0:
            print(f"Processed {count} Steam Spy apps")
        time.sleep(0.1)  # small delay to avoid rate limits

    steamspy_df = pd.DataFrame(steamspy_list)
    print(f"Steam Spy DataFrame generation finished")

    # Step 4: Merge Steam API metadata with Steam Spy stats
    merged_df = pd.merge(steam_df, steamspy_df, on='appid', how='inner')

    # Step 5: Save to CSV
    merged_df.to_csv("steam_games_combined.csv", index=False)
    
if __name__ == "__main__":
    scrape_steam_data()