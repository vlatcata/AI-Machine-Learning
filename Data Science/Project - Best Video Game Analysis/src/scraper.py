import requests
import pandas as pd
import time

def scrape_steam_data():
    """Scrape data from Steam API and Steam Spy.

    Returns:
        pd.DataFrame: A CSV DataFrame file 'steam_games_combined.csv' containing combined data from Steam API and Steam Spy.
    """
    print("Starting to scrape Steam data")
    # Get all Steam apps
    def _get_steam_apps():
        """Fetch all Steam apps from the API.

        Returns:
            pd.DataFrame: A DataFrame containing all Steam apps.
        """
        steam_apps = []
        count = 0
        
        resp = requests.get("https://api.steampowered.com/ISteamApps/GetAppList/v2/")
        apps = resp.json().get('applist', {}).get('apps', [])
        
        for app in apps:
            try:
                r = requests.get(f"https://store.steampowered.com/api/appdetails?appids={app['appid']}", timeout=10)
                if r.status_code != 200:
                    continue
                response_json = r.json()
                app_data = response_json.get(str(app['appid']), {})
                
                if app_data.get('success') and app_data.get('data', {}).get('type') == 'game':
                    steam_apps.append(app_data['data'])
                    count += 1
                    if count % 10 == 0:
                        print(f"Processed {count} Steam apps")
            except Exception as e:
                print(f"Error with appid {app['appid']}: {e}")
                continue

            time.sleep(0.5) # small delay to avoid rate limits

        return pd.DataFrame(steam_apps)

    steam_df = _get_steam_apps()
    steam_df.rename(columns={'steam_appid': 'appid'}, inplace=True)
    print(f"Steam apps DataFrame generation finished")

    # Function to get Steam Spy data
    def _get_steamspy_data(appid):
        """Fetch Steam Spy data for a specific app.

        Args:
            appid (int): The Steam app ID.

        Returns:
            pd.DataFrame: A DataFrame containing the Steam Spy data for the app.
        """
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

    # Loop over apps and collect Steam Spy data (only real games)
    steamspy_list = []
    count = 0

    for appid in steam_df['appid']:
        data = _get_steamspy_data(appid)
        if data:
            steamspy_list.append(data)
        count += 1
        if count % 100 == 0:
            print(f"Processed {count} Steam Spy apps")
        time.sleep(0.5)  # small delay to avoid rate limits

    steamspy_df = pd.DataFrame(steamspy_list)
    print(f"Steam Spy DataFrame generation finished")

    # Merge Steam API metadata with Steam Spy stats
    merged_df = pd.merge(steam_df, steamspy_df, on='appid', how='inner')

    # Save to CSV
    merged_df.to_csv("../data/steam_games_combined.csv", index=False)
    
if __name__ == "__main__":
    scrape_steam_data()