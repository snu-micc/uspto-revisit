import logging
import asyncio
from chemspipy import ChemSpider
from requests.exceptions import RequestException

# List of API keys for ChemSpider
API_KEYS = ["5Ta2mLMsRL4pxj88YjQi03X6iWA0XdmL","mWN307VLtPcyNnWW"]  # Change to your own API keys (list of API keys)

# Global variables to track the current API key and request count
current_key_index = 0
request_count = 0
REQUEST_LIMIT = 980  # Maximum requests per API key

def get_current_api_key():
    """
    Returns the current API key and switches to the next key once the request limit is reached.
    """
    global current_key_index, request_count

    if request_count >= REQUEST_LIMIT:
        # Switch to the next API key
        current_key_index += 1
        request_count = 0  # Reset the request count

        if current_key_index >= len(API_KEYS):
            current_key_index = 0  # Loop back to the first API key if we run out of keys

    return API_KEYS[current_key_index]

def get_smiles_from_ChemSpider(compound_name):
    """
    Fetches the SMILES string for a given compound name from ChemSpider.

    Parameters:
        compound_name (str): The name of the compound to search for.

    Returns:
        tuple: A tuple containing the SMILES string and source ('ChemSpider'), or (None, None) if an error occurs.
    """
    global request_count

    api_key = get_current_api_key()
    cs = ChemSpider(api_key)

    try:
        results = cs.search(compound_name)
        request_count += 1  # Increment the request count after a successful request

        if results:
            return results[0].smiles, 'ChemSpider'
        else:
            logging.error(f"[ChemSpider] No results found for {compound_name}")
    except RequestException as re:
        logging.error(f"[ChemSpider] Network-related error for {compound_name}: {re}")
    except KeyError as ke:
        logging.error(f"[ChemSpider] KeyError when processing results for {compound_name}: {ke}")
    except Exception as e:
        logging.error(f"[ChemSpider] Unexpected error occurred for {compound_name}: {e}")

    return None, None

async def get_smiles_from_ChemSpider_async(compound_name):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_smiles_from_ChemSpider, compound_name)