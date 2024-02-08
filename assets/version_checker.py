import requests

def obtain_tag_name():
    url = 'https://api.github.com/repos/IAHispano/Applio/releases/latest'

    try:
        response = requests.get(url)
        response.raise_for_status()  

        data = response.json()
        tag_name = data['tag_name']

        return tag_name

    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None

def compare_version():
    local_version = "3.0.5"
    online_version = obtain_tag_name()
    elements_online_version = list(map(int, online_version.split('.')))
    elements_local_version = list(map(int, local_version.split('.')))

    for online, local in zip(elements_online_version, elements_local_version):
        if local < online:
            return f"Your local {local_version} version is older than {online_version} the latest version"
        
    return f"Your local version {local_version} is the latest version."
