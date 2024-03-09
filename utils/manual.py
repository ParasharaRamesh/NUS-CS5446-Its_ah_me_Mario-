import pickle
import os

from utils.constants import PLAY_SAVE_LOCATION


def retrieve_play(location):
    '''
    Retrieves a play from a pickle file

    :param location:
    :return:
    '''
    with open(location, 'rb') as f:
        loaded_data = pickle.load(f)

    print(f"Retrieved play stored as pickle file from {location}!")
    return loaded_data

def save_play(buffer, level, agent):
    '''
    Saves an entire play as a pickle file, ensures that every play is saved as a new attempt in the appropriate directory

    :param buffer:
    :param level:
    :param agent:
    :return:
    '''
    levelwise_experience_store_dir = os.path.join(PLAY_SAVE_LOCATION, agent, level)

    if not os.path.exists(levelwise_experience_store_dir):
        os.makedirs(levelwise_experience_store_dir)
        print(f"{levelwise_experience_store_dir} directory did not exist before! Created one for it now..")

    experiences = sorted(os.listdir(levelwise_experience_store_dir))
    if not experiences or len(experiences) == 0:
        file = os.path.join(levelwise_experience_store_dir, "0.pkl")
    else:
        last_file = experiences[-1]
        no = int(last_file.split(".")[0])
        file = os.path.join(levelwise_experience_store_dir, f"{no + 1}.pkl")

    with open(file, 'wb') as f:
        pickle.dump(buffer, f)

    print(f"Saved the play as pickle dump in location {file}")
